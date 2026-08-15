# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""CUDA-graph accelerated pre-deform step of the propagation loop.

Captures everything between two backbone replays that is graph-safe: the two
feature warps (flow_warp), the cond/fp2in cats, and the offset/mask chain
(conv_offset + chunk/tanh/flip/repeat/sigmoid). torchvision's deform_conv2d
is not cudagraph-safe (NaN under capture) and stays eager, consuming the
graph's offset/mask/fp2in outputs directly (same-stream ordering makes the
eager read safe without a clone).
"""

import torch

from lada.models.basicvsrpp.cudagraph_propagate import _BRANCH_ORDER
from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import flow_warp


class OffsetGraphs:
    def __init__(self, net, n, h, w):
        self.net = net
        self.n = self.H = self.W = None
        self.graphs = {}
        self._build(n, h, w)

    def supports(self, n, h, w):
        return self.n == n and self.H == h and self.W == w

    def _build(self, n, h, w):
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype
        C = self.net.mid_channels
        self.graphs = {}
        for name in _BRANCH_ORDER:
            da = self.net.deform_align[name]
            mk = lambda: torch.empty(n, C, h, w, device=device, dtype=dtype)
            s_fp, s_fc, s_fn2 = mk(), mk(), mk()
            s_f1 = torch.empty(n, 2, h, w, device=device, dtype=dtype)
            s_f2 = torch.empty(n, 2, h, w, device=device, dtype=dtype)

            def step(fp, fc, fn2, f1, f2):
                cond_n1 = flow_warp(fp, f1.permute(0, 2, 3, 1))
                cond_n2 = flow_warp(fn2, f2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, fc, cond_n2], dim=1)
                fp2in = torch.cat([fp, fn2], dim=1)
                offset, mask = da.compute_offset_mask(cond, f1, f2)
                return offset, mask, fp2in

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                s_offset, s_mask, s_fp2in = step(s_fp, s_fc, s_fn2, s_f1, s_f2)
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    s_offset, s_mask, s_fp2in = step(s_fp, s_fc, s_fn2, s_f1, s_f2)

            self.graphs[name] = dict(g=g, s_fp=s_fp, s_fc=s_fc, s_fn2=s_fn2,
                                     s_f1=s_f1, s_f2=s_f2, s_offset=s_offset,
                                     s_mask=s_mask, s_fp2in=s_fp2in)
        self.n, self.H, self.W = n, h, w

    def compute(self, module_name, feat_prop, feat_current, feat_n2, flow_n1, flow_n2):
        g = self.graphs[module_name]
        g['s_fp'].copy_(feat_prop)
        g['s_fc'].copy_(feat_current)
        g['s_fn2'].copy_(feat_n2)
        g['s_f1'].copy_(flow_n1)
        g['s_f2'].copy_(flow_n2)
        g['g'].replay()
        # No clone: the caller runs the eager deform_conv2d on the same CUDA
        # stream before the next replay overwrites these buffers.
        return g['s_offset'], g['s_mask'], g['s_fp2in']
