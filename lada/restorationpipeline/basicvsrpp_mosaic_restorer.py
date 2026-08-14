import logging

import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor

logger = logging.getLogger(__name__)

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        # Pre-build the CUDA-graph propagate at load time: capture needs the
        # default stream to itself, so it must not happen inside the worker
        # threads. Feature maps are always 64x64 for 256x256 clips.
        gen = getattr(self.model, "generator_ema", None) or self.model.generator
        prepare = getattr(gen, "prepare_cudagraph_propagate", None)
        if prepare is not None:
            try:
                prepare(1, 64, 64)
            except Exception as e:
                logger.warning("CUDA-graph propagate pre-build failed: %s", e)
        prepare_up = getattr(gen, "prepare_cudagraph_upsample", None)
        if prepare_up is not None:
            try:
                prepare_up(1, 64, 64)
            except Exception as e:
                logger.warning("CUDA-graph upsample pre-build failed: %s", e)
        prepare_off = getattr(gen, "prepare_cudagraph_offsets", None)
        if prepare_off is not None:
            try:
                prepare_off(1, 64, 64)
            except Exception as e:
                logger.warning("CUDA-graph offsets pre-build failed: %s", e)

    def restore(self, video: list[ImageTensor], max_frames=-1) -> list[ImageTensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            # build the (T, C, H, W) stack in contiguous (NCHW) layout; the
            # model and its CUDA graphs capture in NCHW to avoid the numeric
            # corruption channels_last introduces in the recurrent propagate.
            inference_view = torch.stack([x.permute(2, 0, 1) for x in video], dim=0).to(device=self.device).to(dtype=self.dtype).div_(255.0).unsqueeze(0)

            if max_frames > 0:
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self.model(inputs=inference_view[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self.model(inputs=inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
            result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
