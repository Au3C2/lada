# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""CUDA-graph accelerated deform-alignment offset/mask computation.

The offsets chain (cat + conv_offset + chunk/tanh/flip/repeat/sigmoid) is a
pure conv/elementwise pipeline, so it captures bit-exactly and removes ~14
kernel launches per frame-step. torchvision's deform_conv2d itself is not
cudagraph-safe and stays eager, consuming the graph's offset/mask outputs
directly (same-stream ordering makes the eager read safe without a clone).
"""

import torch

from lada.models.basicvsrpp.cudagraph_propagate import _BRANCH_ORDER


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
            s_cond = torch.empty(n, 3 * C, h, w, device=device, dtype=dtype)
            s_f1 = torch.empty(n, 2, h, w, device=device, dtype=dtype)
            s_f2 = torch.empty(n, 2, h, w, device=device, dtype=dtype)

            def step(cond, f1, f2):
                return da.compute_offset_mask(cond, f1, f2)

            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                s_offset, s_mask = step(s_cond, s_f1, s_f2)
                torch.cuda.synchronize()
                with torch.cuda.graph(g):
                    s_offset, s_mask = step(s_cond, s_f1, s_f2)

            self.graphs[name] = dict(g=g, s_cond=s_cond, s_f1=s_f1,
                                     s_f2=s_f2, s_offset=s_offset, s_mask=s_mask)
        self.n, self.H, self.W = n, h, w

    def compute(self, module_name, cond, flow_n1, flow_n2):
        g = self.graphs[module_name]
        g['s_cond'].copy_(cond)
        g['s_f1'].copy_(flow_n1)
        g['s_f2'].copy_(flow_n2)
        g['g'].replay()
        # No clone: the caller runs the eager deform_conv2d on the same CUDA
        # stream before the next replay overwrites these buffers.
        return g['s_offset'], g['s_mask']
