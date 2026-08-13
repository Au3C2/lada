# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""CUDA-graph accelerated propagate for BasicVSR++.

The per-frame backbone (cat + residual chain + add) is launch-bound at 64x64
feature resolution; capturing it as one CUDA graph and replaying it removes ~30
kernel launches per frame-step. deform_align and the flow warps stay eager:
torchvision's deform_conv2d is not cudagraph-safe, so including it in a graph
would change the numerics. Because every captured op replays bit-identically
and everything else stays eager, the output is bit-identical to the eager path.

Measured ~1.9x faster generator forward (T=113 clip) with max diff 0.0.
"""

import torch

from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import flow_warp

_BRANCH_ORDER = ['backward_1', 'forward_1', 'backward_2', 'forward_2']


class PropagateGraphs:
    def __init__(self, net, n, h, w):
        self.net = net
        self.C = net.mid_channels
        self.n = self.H = self.W = None
        self.graphs = {}
        self._build(n, h, w)

    def supports(self, n, h, w):
        return self.n == n and self.H == h and self.W == w

    @staticmethod
    def _other_names(module_name):
        i = _BRANCH_ORDER.index(module_name)
        return [k for k in _BRANCH_ORDER if _BRANCH_ORDER.index(k) < i]

    def _build(self, n, h, w):
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype
        self.graphs = {}
        for name in _BRANCH_ORDER:
            bb = self.net.backbone[name]
            others = self._other_names(name)
            def mk():
                return torch.empty(n, self.C, h, w, device=device, dtype=dtype,
                                   memory_format=torch.channels_last)
            s_fc, s_fp = mk(), mk()
            s_others = [mk() for _ in others]

            def bb_step(fc, fp, os_):
                return fp + bb(torch.cat([fc] + os_ + [fp], dim=1))

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                bb_step(s_fc, s_fp, s_others)
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    s_out = bb_step(s_fc, s_fp, s_others)

            self.graphs[name] = dict(s_fc=s_fc, s_fp=s_fp, s_others=s_others,
                                     others=others, g=g, s_out=s_out)
        self.n, self.H, self.W = n, h, w

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()
        g = self.graphs[module_name]
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.C, h, w)
        da = self.net.deform_align[module_name]
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if i > 1:
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    feat_n2 = feats[module_name][-2]
                else:
                    flow_n2 = torch.zeros_like(flow_n1)
                    feat_n2 = torch.zeros_like(feat_prop)
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                fp2in = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = da(fp2in, cond, flow_n1, flow_n2)
            g['s_fc'].copy_(feat_current)
            g['s_fp'].copy_(feat_prop)
            for s, v in zip(g['s_others'], [feats[k][idx] for k in g['others']]):
                s.copy_(v)
            g['g'].replay()
            feat_prop = g['s_out'].clone()
            feats[module_name].append(feat_prop)
        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats


class UpsampleGraphs:
    """CUDA-graph for the per-frame reconstruction/upsample step.

    The reconstruction (cat 5 branches -> residual blocks -> 2x pixel-shuffle ->
    final convs -> residual add of the input) is a straight-line conv chain, so
    it captures bit-exactly. The 5-way feature cat is done eagerly into a static
    buffer; lqs[:, i] is copied in per frame. Output is bit-identical to eager.
    """

    def __init__(self, net, n, h, w):
        self.net = net
        self.n = n
        self.H, self.W = h, w
        self.C = net.mid_channels
        device = next(net.parameters()).device
        dtype = next(net.parameters()).dtype
        ih, iw = 4 * h, 4 * w  # input lqs spatial
        s_hr = torch.empty(n, 5 * self.C, h, w, device=device, dtype=dtype,
                           memory_format=torch.channels_last)
        s_lq = torch.empty(n, 3, ih, iw, device=device, dtype=dtype,
                           memory_format=torch.channels_last)
        self.s_hr = s_hr
        self.s_lq = s_lq

        def step(hr, lq):
            hr = net.reconstruction(hr)
            hr = net.lrelu(net.upsample1(hr))
            hr = net.lrelu(net.upsample2(hr))
            hr = net.lrelu(net.conv_hr(hr))
            hr = net.conv_last(hr)
            return hr + lq

        self.g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            step(s_hr, s_lq)
            torch.cuda.synchronize()
            with torch.cuda.graph(self.g):
                self.s_out = step(s_hr, s_lq)

    def supports(self, n, h, w):
        return self.n == n and self.H == h and self.W == w

    def upsample(self, lqs, feats):
        n, t, c, h, w = lqs.size()
        order = [k for k in feats if k != 'spatial']
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]
        outputs = []
        for i in range(t):
            hr = [feats[k].pop(0) for k in order]
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            self.s_hr.copy_(torch.cat(hr, dim=1))
            self.s_lq.copy_(lqs[:, i, :, :, :])
            self.g.replay()
            outputs.append(self.s_out.clone())
        return torch.stack(outputs, dim=1)
