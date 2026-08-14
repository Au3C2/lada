# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import csv
import dataclasses
import io
import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from typing import Callable, Iterator, Tuple, Literal
from collections import deque
import heapq
import shlex
import threading
from queue import Full, Queue

import av
import cv2
import torch
import numpy as np

from lada.utils import Image, Mask, VideoMetadata, os_utils
from lada.utils.threading_utils import STOP_MARKER

logger = logging.getLogger(__name__)

def read_video_frames(path: str, float32: bool = True, start_idx: int = 0, end_idx: int | None = None, normalize_neg1_pos1 = False, binary_frames=False) -> list[np.ndarray]:
    with VideoReaderOpenCV(path) as video_reader:
        frames = []
        i = 0
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if ret and (end_idx is None or i < end_idx):
                if binary_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)
                if i >= start_idx:
                    if float32:
                        if normalize_neg1_pos1:
                            frame = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
                        else:
                            frame = frame.astype(np.float32) / 255.
                    frames.append(frame)
                i += 1
            else:
                break
    return frames

def resize_video_frames(frames: list, size: int | tuple[int, int]):
    resized = []
    target_size = size if isinstance(size, (list, tuple)) else (size, size)
    for frame in frames:
        if frame.shape[:2] == target_size:
            resized.append(frame)
        else:
            resized.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    return resized

def pad_to_compatible_size_for_video_codecs(imgs):
    # dims need to be divisible by 2 by most codecs. given the chroma / pix format dims must be divisible by 4
    h, w = imgs[0].shape[:2]
    pad_h = 0 if h % 4 == 0 else 4 - (h % 4)
    pad_w = 0 if w % 4 == 0 else 4 - (w % 4)
    if pad_h == 0 and pad_w == 0:
        return imgs
    else:
        return [np.pad(img, ((0, pad_h), (0, pad_w), (0,0))).astype(np.uint8) for img in imgs]

@contextmanager
def VideoReaderOpenCV(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file:", *args)
    try:
        yield cap
    finally:
        cap.release()

class VideoReader:
    def __init__(self, file):
        self.file = file
        self.container = None

    def __enter__(self):
        # We currently do not pass through metadata to the output file so let's just ignore potential errors. Fixes #127
        # E.g. metadata could be encoded in CP936 instead of UTF-8 which would raise an error if we don't pass it in metadata_encoding.
        # If we use it in the future we have to consider non-default character encodings.
        self.container = av.open(self.file, metadata_errors='ignore')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def frames(self) -> Iterator[Tuple[torch.Tensor, int]]:
        # Print to console via FFmpegs log callback instead of utilizing Pythons logging system
        # Unfortunately we need this to prevent deadlocks. On certain corrupt video files decode() would hang indefinitely after
        # encountering an error (always reproducible). See https://github.com/PyAV-Org/PyAV/issues/751 and https://codeberg.org/ladaapp/lada/issues/247
        # Alternatively, setting thread_type to 'SLICE' would also avoid the deadlock even with av logs enabled but may negatively impact performance.
        av.logging.restore_default_callback()
        av.logging.set_libav_level(av.logging.ERROR)
        self.container.streams.video[0].thread_type = 'AUTO'

        # Fault-tolerant frame decoding with frame duplication for corrupted frames
        # This approach mimics how ffmpeg CLI handles corrupted frames by duplicating the last good frame
        last_good_frame = None
        consecutive_errors = 0
        max_consecutive_errors = 10  # Prevent infinite loops on completely corrupted streams
        
        # Use packet-level decoding to handle corrupted frames properly
        vstream = self.container.streams.video[0]
        for packet in self.container.demux(vstream):
            try:
                frames = packet.decode()
                for frame in frames:
                    nd_frame = frame.to_ndarray(format='bgr24')
                    torch_frame = torch.from_numpy(nd_frame)
                    last_good_frame = (torch_frame, frame.pts)
                    consecutive_errors = 0
                    yield torch_frame, frame.pts
            except av.error.InvalidDataError as e:
                # Handle corrupted frames by duplicating the last good frame
                if last_good_frame is not None and consecutive_errors < max_consecutive_errors:
                    consecutive_errors += 1
                    logger.warning(f"Corrupted frame detected, duplicating last good frame ({consecutive_errors}/{max_consecutive_errors})")
                    yield last_good_frame[0], last_good_frame[1]
                else:
                    # No good frame available yet (first frame corrupt) or too many consecutive errors
                    # Re-raise the error instead of skipping to fail fast
                    raise Exception(f"Cannot handle corrupted frame: {'first frame corrupt' if last_good_frame is None else f'too many consecutive corrupted frames ({max_consecutive_errors})'}") from e
            except Exception as e:
                # For other unexpected errors, re-raise them
                raise

    def seek(self, offset_ns):
        offset = int((offset_ns / 1_000_000_000) * av.time_base)
        self.container.seek(offset)

def get_video_meta_data(path: str) -> VideoMetadata:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    out, err =  p.communicate()
    if p.returncode != 0:
        raise Exception(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    value = [int(num) for num in json_video_stream['avg_frame_rate'].split("/")]
    # Can be 0/0 for some files for ffprobe isn't able to determine the number of frames nb_frames
    average_fps = value[0]/value[1] if len(value) == 2 and value[1] != 0 else value[0]

    value = [int(num) for num in json_video_stream['r_frame_rate'].split("/")]
    fps = value[0]/value[1] if len(value) == 2 else value[0]
    fps_exact = Fraction(value[0], value[1])

    value = [int(num) for num in json_video_stream['time_base'].split("/")]
    time_base = Fraction(value[0], value[1])

    frame_count = json_video_stream.get('nb_frames')
    if not frame_count:
        # print("frame count ffmpeg", frame_count)
        cap = cv2.VideoCapture(path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        # print("frame count opencv", frame_count)
    frame_count=int(frame_count)

    start_pts = json_video_stream.get('start_pts')

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        frames_count=frame_count,
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts
    )
    return metadata

def offset_ns_to_frame_num(offset_ns, video_fps_exact):
    return int(Fraction(offset_ns, 1_000_000_000) * video_fps_exact)

def write_frames_to_video_file(frames: list[Image], output_path, fps: int | float | Fraction, codec='x264', preset='medium', crf=None):
    assert frames[0].ndim == 3
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-preset', preset
    ]
    if codec == 'x265':
        ffmpeg_output.extend(['-tag:v', 'hvc1', '-vcodec', 'libx265', '-crf', str(crf) if crf else '18'])
    elif codec == 'x264':
        ffmpeg_output.extend(['-vcodec', 'libx264', '-crf', str(crf) if crf else '15'])
    ffmpeg_output.append(output_path)

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")

def write_masks_to_video_file(frames: list[Mask], output_path, fps: int | float | Fraction):
    #assert frames[0].ndim == 2
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-vcodec', 'ffv1', '-level', '3', '-tag:v', 'ffv1',  output_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"ERROR when writing video via ffmpeg to file: {output_path}")
            print(f"exception: {e}")
            print(f"stderr: {ffmpeg_process.stderr.read()}")
            print(f"stdout: {ffmpeg_process.stdout.read()}")
            raise e
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")
        print(f"stdout: {ffmpeg_process.stdout.read()}")

def process_video_v3(input_path, output_path, frame_processor: Callable[[Image], Image]):
    video_metadata = get_video_meta_data(input_path)
    video_reader = cv2.VideoCapture(input_path)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps=video_metadata.video_fps, frameSize=(video_metadata.video_width, video_metadata.video_height))
    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if ret:
            processed_frame = frame_processor(frame)
            video_writer.write(processed_frame)
        else:
            break
    video_reader.release()
    video_writer.release()

def approx_memory(video_metadata: VideoMetadata, frames_count, assume_images=True, assume_masks=True):
    size = 0
    frame_size_image = video_metadata.video_width * video_metadata.video_height * 3 * 1
    frame_size_mask = video_metadata.video_width * video_metadata.video_height * 1 * 1
    if assume_images:
        size += frame_size_image * frames_count
    if assume_masks:
        size += frame_size_mask * frames_count
    return size

def approx_max_length_by_memory_limit(video_metadata: VideoMetadata, limit_in_megabytes, assume_images=True, assume_masks=True):
    frame_size_image = approx_memory(video_metadata, 1, assume_images=assume_images, assume_masks=assume_masks)
    max_length_frames = (limit_in_megabytes * 1024 * 1024) / frame_size_image
    max_length_seconds = int(max_length_frames / video_metadata.video_fps)
    return max_length_seconds

@dataclass
class EncodingPreset:
    name: str
    description: str
    user_preset: bool
    encoder_name: str
    encoder_options: str

    def __hash__(self): return hash(self.name)

    def clone(self): return EncodingPreset(**dataclasses.asdict(self))

def get_default_preset_name():
    if os_utils.has_nvidia_gpu() and is_nvidia_cuda_encoding_available():
        return "hevc-nvidia-gpu-hq"
    if is_apple_videotoolbox_encoding_available():
        return "hevc-apple-gpu-balanced"
    if os_utils.has_intel_arc_gpu() and is_intel_qsv_encoding_available():
        return "hevc-intel-gpu-hq"
    return "h264-cpu-fast"

@cache
def get_encoding_presets() -> list[EncodingPreset]:
    presets = []
    encoding_presets_csv_path = os.path.join(os.path.dirname(__file__), 'encoding_presets.csv')
    if not os.path.exists(encoding_presets_csv_path):
        logger.warning("Could not find encoding_presets.csv!")
        return presets
    
    available_encoders_list = get_video_encoder_codecs()
    available_encoder_names = {e.name.lower() for e in available_encoders_list}
    has_intel_qsv = False
    if 'h264_qsv' in available_encoder_names:
        has_intel_qsv = is_intel_qsv_encoding_available()
    has_nvidia_nvenc = False
    if 'h264_nvenc' in available_encoder_names:
        has_nvidia_nvenc = is_nvidia_cuda_encoding_available()
    has_apple_vt = False
    if 'hevc_videotoolbox' in available_encoder_names or 'h264_videotoolbox' in available_encoder_names:
        has_apple_vt = is_apple_videotoolbox_encoding_available()

    with open(encoding_presets_csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='|')
        for row in reader:
            encoder_name = row["encoder_name"].lower()
            preset_name = row["preset_name"].lower()
            
            if encoder_name not in available_encoder_names:
                continue

            # Nvidia
            is_nvidia_preset = 'nvenc' in encoder_name or 'nvidia' in preset_name
            if is_nvidia_preset and not has_nvidia_nvenc:
                continue
            # Intel
            is_intel_preset = 'qsv' in encoder_name or 'intel' in preset_name
            if is_intel_preset and not has_intel_qsv:
                continue
            # Apple Video Toolbox
            is_apple_preset = 'videotoolbox' in encoder_name or 'apple' in preset_name
            if is_apple_preset and not has_apple_vt:
                continue

            preset = EncodingPreset(row["preset_name"], row["preset_description(translatable)"], False, row["encoder_name"], row["encoder_options"])    
            presets.append(preset)
        return presets

# ffmpeg nvenc options that take no value (bare flags). lada's presets only use the
# first two, but users may type any of these into the GUI's custom-options field.
_PYNC_FLAG_OPTIONS = {
    "temporal_aq", "temporal-aq", "2pass", "zerolatency", "strict_gop", "no-scenecut",
    "intra-refresh", "single-slice-intra-refresh", "forced-idr", "nonref_p", "aud",
    "a53cc", "udu_sei", "extra_sei", "constrained-encoding", "rgb_mode",
}

# ffmpeg nvenc options with no PyNvVideoCodec equivalent. They are silently dropped
# (PyNvVideoCodec.CreateEncoder raises on unknown options) and reported by
# translate_nvenc_encoder_options_to_pync() so the caller can warn the user.
_PYNC_UNSUPPORTED_OPTIONS = {
    "b_ref_mode", "level", "tier", "aud", "a53cc", "udu_sei", "extra_sei",
    "max_slice_size", "split_encode_mode", "intra-refresh", "single-slice-intra-refresh",
    "forced-idr", "nonref_p", "strict_gop", "no-scenecut", "refs", "weighted_pred",
    "dpb_size", "unidir_b", "constrained-encoding", "rgb_mode", "surfaces", "gpu",
    "qblur", "qcomp", "qdiff", "lookahead_level", "weighted_pred", "zerolatency",
}


def _normalize_bitrate(value: str) -> str:
    """Convert an ffmpeg bitrate string (e.g. ``800k``, ``4M``, ``2G``, ``3Mi``) to
    plain bits-per-second, which is what PyNvVideoCodec's -bitrate/-maxbitrate/-vbvbufsize accept."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgiKMGI]?[gG]?)?", value)
    if not match:
        return value
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    multiplier = {"": 1, "k": 1000, "m": 1000000, "g": 1000000000,
                  "ki": 1024, "mi": 1024 * 1024, "gi": 1024 * 1024 * 1024}.get(unit, 1)
    return str(int(number * multiplier))


def translate_nvenc_encoder_options_to_pync(encoder_name: str, encoder_options: str) -> tuple[dict[str, str], list[str]]:
    """Translate an ffmpeg/PyAV NVENC option string into PyNvVideoCodec CreateEncoder() kwargs.

    ``encoder_options`` is the option string lada stores in ``encoding_presets.csv`` (or the
    string a user typed into the GUI's custom-options field while ``encoder_name`` is
    ``h264_nvenc``/``hevc_nvenc``). The returned dict is meant to be forwarded as
    ``**kwargs`` to ``PyNvVideoCodec.CreateEncoder(width, height, format, True, **kwargs)``.

    Returns ``(pync_kwargs, dropped)`` where ``dropped`` lists every ffmpeg option that has
    no PyNvVideoCodec equivalent and was therefore omitted.

    Alignment coverage
    ==================
    Options translated 1:1 (behaviour matches ffmpeg):

    ====================== =========================================================
    ffmpeg option          PyNvc kwarg
    ====================== =========================================================
    ``-preset p1..p7``     ``preset="P1..P7"``
    ``-tune hq/uhq/ll/ull/lossless``  ``tuning_info`` (``high_quality``/``uhq``/``low_latency``/``ultra_low_latency``/``lossless``)
    ``-rc constqp/vbr/cbr`` ``rc``
    ``-qp N``              ``constqp="N,B,I"`` (see QP note below)
    ``-bf N``              ``bf``
    ``-spatial_aq 1 -aq-strength S`` ``aq=S``
    ``-temporal_aq``       ``temporalaq`` (bare flag on both sides)
    ``-rc-lookahead N``    ``lookahead``
    ``-g N``               ``gop``
    ``-multipass disabled/qres/fullres``, ``-2pass`` ``multipass``
    ``-profile``           ``profile`` (hevc ``rext`` -> ``frext``)
    ``-b``/``-maxrate``/``-bufsize`` ``bitrate``/``maxbitrate``/``vbvbufsize`` (units normalised)
    ``-init_qpP/-init_qpI/-init_qpB`` ``initqp="P,B,I"``
    ``-qmin``/``-qmax``    ``qmin``/``qmax``
    ``-cq``/``-quality``   ``cq`` (VBR target quality)
    ====================== =========================================================

    Options with no PyNvVideoCodec equivalent are dropped and reported in ``dropped``:
    ``-b_ref_mode`` (PyNvc always uses ``BFRAME_REF_MODE_DISABLED``; lada's own hevc
    presets pass ``-b_ref_mode middle``), ``-level``, ``-tier``, ``-aud``, ``-a53cc``,
    ``-udu_sei``, ``-extra_sei``, ``-max_slice_size``, ``-intra-refresh``,
    ``-forced-idr``, ``-zerolatency``, ``-strict_gop``, ``-refs``, ``-weighted_pred``
    and other low-level flags.

    ``-qp N`` QP mapping note
    -------------------------
    ffmpeg's ``-qp`` only sets the P-frame QP to ``N``; the I and B QPs are derived from
    libavcodec's ``i_quant_factor``/``b_quant_factor``/``b_quant_offset`` defaults
    (``-0.8``, ``1.25``, ``1.25``) and assigned to NVENC with C float->int truncation:
    ``I = int(N*0.8 + 0.5)`` and ``B = int(N*1.25 + 1.75)``.  This function reproduces
    exactly those QPs, so the encoder configuration matches ``ffmpeg -c:v *_nvenc``.
    Output pixels still differ slightly because the two backends encode independently
    at the same QP (SSIM ~0.99); the inputs they feed NVENC are aligned since
    :class:`PyncVideoWriter` now converts BGR to limited-range NV12 (the same signal
    the PyAV backend feeds via ``yuv420p``).

    Remaining size caveats
    ----------------------
    - Both backends encode limited-range YUV, so file sizes for identical QPs match
      closely (~1 %).
    - ``-b_ref_mode``/``-level``/``-tier`` are dropped, so unusual profiles or B-ref
      structures are not reproduced.
    - Everything above assumes the codec string in ``encoder_name`` is a valid NVENC
      codec; otherwise ``codec`` is omitted from the result.

    Example
    -------
    .. code-block:: python

        import PyNvVideoCodec as nvc

        # lada preset 'hevc-nvidia-gpu-uhq' (from encoding_presets.csv):
        kwargs, dropped = translate_nvenc_encoder_options_to_pync(
            "hevc_nvenc",
            "-preset p7 -tune hq -rc constqp -qp 18 -spatial_aq 1 -aq-strength 6 "
            "-bf 3 -b_ref_mode middle -rc-lookahead 32",
        )
        # kwargs == {'codec': 'hevc', 'preset': 'P7', 'tuning_info': 'high_quality',
        #             'rc': 'constqp', 'constqp': '18,33,15', 'bf': '3', 'aq': '6',
        #             'lookahead': '32'}
        # dropped == ['-b_ref_mode middle']   # no PyNvc equivalent; warn the user

        encoder = nvc.CreateEncoder(1920, 1080, "NV12", True, gpu_id="0", **kwargs)
        muxer = nvc.FFmpegMuxer(
            file_path=output_path, media_format=nvc.GetMediaFormat(output_path),
            codec="hevc", width=1920, height=1080, fps_num=30, fps_den=1,
            timebase_num=1, timebase_den=90000, extradata=encoder.GetSequenceParams(),
        )
        muxer.SetUniformPtsIncrement(3000)
        # ... encode BGR frames converted to NV12 (limited-range YUV), mux each packet ...
        muxer.Finalize()
        del muxer  # REQUIRED: FFmpegMuxer only flushes to disk in its destructor
    """
    tokens = shlex.split(encoder_options)
    parsed: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("-"):
            i += 1
            continue
        name = token.lstrip("-").lower()
        if name in _PYNC_FLAG_OPTIONS:
            parsed[name] = ""
            i += 1
        elif i + 1 < len(tokens):
            parsed[name] = tokens[i + 1]
            i += 2
        else:
            logger.warning("Ignoring option without a value: %s", token)
            i += 1

    dropped: list[str] = []
    out: dict[str, str] = {}

    codec = "hevc" if "hevc" in encoder_name.lower() else ("h264" if "h264" in encoder_name.lower() else None)
    if codec:
        out["codec"] = codec

    if "preset" in parsed:
        out["preset"] = parsed["preset"].upper()

    tune_map = {
        "hq": "high_quality", "uhq": "uhq", "ll": "low_latency",
        "ull": "ultra_low_latency", "lossless": "lossless",
    }
    if "tune" in parsed:
        tune = parsed["tune"].lower()
        if tune in tune_map:
            out["tuning_info"] = tune_map[tune]
        else:
            dropped.append(f"-tune {parsed['tune']}")

    if "rc" in parsed:
        rc = parsed["rc"].lower()
        rc_normalized = {
            "cbr_hq": "cbr", "cbr_ld_hq": "cbr", "vbr_hq": "vbr",
            "vbr_2pass": "vbr", "vbr_minqp": "vbr",
            "ll_2pass_quality": "vbr", "ll_2pass_size": "vbr",
        }.get(rc, rc)
        if rc_normalized in ("constqp", "vbr", "cbr"):
            out["rc"] = rc_normalized
        else:
            dropped.append(f"-rc {parsed['rc']}")

    if "qp" in parsed:
        try:
            n = int(parsed["qp"])
            # ffmpeg's -qp sets only P; I and B derive from libavcodec defaults
            # (i_quant_factor=-0.8, b_quant_factor=1.25, b_quant_offset=1.25) with the
            # C float->int truncation:  I = int(N*0.8 + 0.5), B = int(N*1.25 + 1.75).
            out["constqp"] = f"{n},{int(n * 1.25 + 1.75)},{int(n * 0.8 + 0.5)}"
        except ValueError:
            dropped.append(f"-qp {parsed['qp']}")
    if "constqp" in parsed:
        out["constqp"] = parsed["constqp"]

    if "bf" in parsed:
        out["bf"] = parsed["bf"]

    # ffmpeg spells it -spatial_aq, but users may type -spatial-aq; both mean "enable
    # spatial AQ", and its strength comes from -aq-strength (ffmpeg default 8).
    spatial_aq = parsed.get("spatial_aq", parsed.get("spatial-aq", "0"))
    if spatial_aq not in ("0", ""):
        out["aq"] = parsed.get("aq-strength", "8")
    elif "aq-strength" in parsed:
        dropped.append("-aq-strength (without -spatial_aq 1 it does nothing in ffmpeg)")

    if "temporal_aq" in parsed or "temporal-aq" in parsed:
        out["temporalaq"] = ""

    if "rc-lookahead" in parsed:
        out["lookahead"] = parsed["rc-lookahead"]

    if "g" in parsed:
        out["gop"] = parsed["g"]

    if "multipass" in parsed:
        mp = parsed["multipass"].lower()
        if mp in ("disabled", "qres", "fullres"):
            out["multipass"] = mp
        else:
            dropped.append(f"-multipass {parsed['multipass']}")
    if "2pass" in parsed:
        out["multipass"] = "fullres"

    if "profile" in parsed:
        out["profile"] = "frext" if parsed["profile"].lower() == "rext" else parsed["profile"].lower()

    # ffmpeg writes -b:v / -maxrate:v / -bufsize:v; strip the stream selector ":v".
    for ffmpeg_name, pync_name in (("b", "bitrate"), ("maxrate", "maxbitrate"), ("bufsize", "vbvbufsize")):
        value = parsed.get(ffmpeg_name, parsed.get(f"{ffmpeg_name}:v"))
        if value is not None:
            out[pync_name] = _normalize_bitrate(value)

    if "init_qpP" in parsed or "init_qpI" in parsed or "init_qpB" in parsed:
        out["initqp"] = f"{int(parsed.get('init_qpP', '0'))},{int(parsed.get('init_qpB', '0'))},{int(parsed.get('init_qpI', '0'))}"

    if "qmin" in parsed:
        out["qmin"] = parsed["qmin"]
    if "qmax" in parsed:
        out["qmax"] = parsed["qmax"]
    if "cq" in parsed:
        out["cq"] = parsed["cq"]
    elif "quality" in parsed:
        out["cq"] = parsed["quality"]

    for name, value in parsed.items():
        if name in out or name in ("constqp",):
            continue
        if name in ("tune", "preset", "rc", "qp", "bf", "spatial_aq", "spatial-aq", "aq-strength",
                    "temporal_aq", "temporal-aq", "rc-lookahead", "g", "multipass", "2pass",
                    "profile", "b", "b:v", "maxrate", "maxrate:v", "bufsize", "bufsize:v",
                    "init_qpP", "init_qpI", "init_qpB", "qmin", "qmax", "cq", "quality"):
            continue
        dropped.append(f"-{name}{' ' + value if value else ''}")

    if dropped:
        logger.warning("PyNvVideoCodec has no equivalent for: %s", "; ".join(dropped))

    return out, dropped


@cache
def is_intel_qsv_encoding_available() -> bool:
    if sys.platform == "win32":
        return _is_codec_hardware_acceleration_working('h264_qsv', 'qsv')
    else:
        # TODO: For some reason the method HWAccel.create() is not working for qsv when using official Linux binary wheel of PyAv 16.1.0
        # It throws a "Function not implemented" error regardless whether qsv is working or not (cuda/nvenc check works as expected)
        # It works when building PyAV locally against ffmpeg from ArchLinux on the same system
        # As a workaround let's encode a dummy frame to see if qsv is working
        # See issue: #297
        try:
            with av.logging.Capture():
                mem_file = io.BytesIO()
                with av.open(mem_file, mode='w', format='mp4') as container:
                    stream = container.add_stream('h264_qsv', rate=30)
                    stream.width = 64
                    stream.height = 64
                    stream.pix_fmt = 'nv12'
                    dummy_frame = av.VideoFrame(64, 64, format='nv12')
                    stream.encode(dummy_frame)
                    return True
        except Exception:
            return False

@cache
def is_nvidia_cuda_encoding_available() -> bool:
    return _is_codec_hardware_acceleration_working('h264_nvenc', 'cuda')

@cache
def is_apple_videotoolbox_encoding_available() -> bool:
    if sys.platform != "darwin":
        return False
    if _is_codec_hardware_acceleration_working('hevc_videotoolbox', 'videotoolbox'):
        return True
    # HWAccel check can fail in frozen/PyInstaller builds. Try encoding a dummy frame instead (same approach as QSV on Linux).
    try:
        with av.logging.Capture():
            mem_file = io.BytesIO()
            with av.open(mem_file, mode='w', format='mp4') as container:
                stream = container.add_stream('hevc_videotoolbox', rate=30)
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = 'yuv420p'
                dummy_frame = av.VideoFrame(64, 64, format='yuv420p')
                stream.encode(dummy_frame)
                return True
    except Exception:
        return False

def _is_codec_hardware_acceleration_working(codec_name: str, hwaccel_device_type: str, codec_mode: Literal["r", "w"]='w') -> bool:
    try:
        with av.logging.Capture():
            hwaccel = av.codec.hwaccel.HWAccel(hwaccel_device_type, allow_software_fallback=False)
            codec = av.codec.Codec(codec_name, codec_mode)
            # Initialize hardware context. This will raise if hardware is not available (missing, driver or library issue)
            hwaccel.create(codec)
            return True
    except Exception:
        return False

def _is_nvenc_encoder(encoder: str) -> bool:
    return encoder.lower() in ("h264_nvenc", "hevc_nvenc")

@cache
def is_pync_encoding_available() -> bool:
    """True if PyNvVideoCodec can be imported (and thus used to speed up NVENC export).

    Note: PyNvVideoCodec's native module depends on CUDA DLLs that torch bundles in its
    own ``lib`` dir; it must therefore be imported *after* ``torch`` has been imported
    (video_utils already imports torch at module level, which registers the DLL path).
    """
    try:
        import torch  # noqa: F401  (registers torch/lib CUDA DLLs on the search path)
        import PyNvVideoCodec  # noqa: F401
        return True
    except Exception:
        return False

class PyncVideoWriter:
    """Video writer backed by PyNvVideoCodec (NVIDIA's low-level GPU encode API).

    Exposes the same ``write()``/``release()``/context-manager interface as
    :class:`VideoWriter`, so the restore/export pipeline can treat both interchangeably.
    Used automatically by :class:`VideoWriter` when the selected encoder is
    ``h264_nvenc``/``hevc_nvenc`` and PyNvVideoCodec is installed (see
    :func:`is_pync_encoding_available`). Any failure during initialisation falls back to
    the PyAV path.

    Behavioural differences vs. the PyAV backend (see also the docstring of
    :func:`translate_nvenc_encoder_options_to_pync`):

    - Input frames (BGR, HxWx3 uint8) are converted to the NV12 buffer format
      (limited-range YUV) in :func:`write` and handed to NVENC as a CPU buffer. The
      conversion produces the same signal as the PyAV backend's own ``bgr24 -> yuv420p``
      step, so both backends feed NVENC the same limited-range YUV.
    - PTS are emitted uniformly (``90000 / fps`` ticks per frame); non-uniform/VFR
      source timing is not preserved.
    - ``mp4_fast_start`` is not supported by the FFmpegMuxer and is ignored.
    - ``-b_ref_mode``/``-level``/``-tier`` and other ffmpeg options without a PyNvc
      equivalent are dropped; the list is logged via :func:`logger.warning`.
    """

    def __init__(self, output_path, width, height, fps, encoder: str, encoder_options: str,
                 time_base=None, mp4_fast_start=False, gpu_id="0"):
        import PyNvVideoCodec as nvc
        self._nvc = nvc
        kwargs, dropped = translate_nvenc_encoder_options_to_pync(encoder, encoder_options)
        if dropped:
            logger.warning("PyNvVideoCodec backend dropping unsupported options: %s", "; ".join(dropped))

        if hasattr(fps, "numerator"):
            fps_num, fps_den = int(fps.numerator), int(fps.denominator)
        else:
            fps_float = float(fps)
            fps_num, fps_den = (int(round(fps_float)), 1) if fps_float.is_integer() else (round(fps_float * 1000), 1000)

        timebase_den = 90000
        pts_inc = max(1, int(round(timebase_den * fps_den / fps_num)))

        self._encoder = nvc.CreateEncoder(width, height, "NV12", True, gpu_id=gpu_id, fps=str(fps_num), **kwargs)
        self._muxer = nvc.FFmpegMuxer(
            file_path=output_path, media_format=nvc.GetMediaFormat(output_path),
            codec=kwargs["codec"], width=width, height=height,
            fps_num=fps_num, fps_den=fps_den, timebase_num=1, timebase_den=timebase_den,
            extradata=self._encoder.GetSequenceParams())
        self._muxer.SetUniformPtsIncrement(pts_inc)
        self._frame_index = 0
        # Persistent conversion buffers: the BGR->NV12 rebuild is a pure-CPU hot path
        # in write(); reusing the arrays avoids ~0.5 ms/frame of numpy allocation
        # (measured on 1080p, ~30% of the conversion subtree).
        self._hw = width * height
        self._nv12 = np.empty(self._hw * 3 // 2, dtype=np.uint8)
        self._uv = np.empty(self._hw // 2, dtype=np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def write(self, frame, frame_pts=None):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            frame = np.ascontiguousarray(frame).astype(np.uint8)
        # BGR (full-range RGB, as decoded by VideoReader) -> NV12 (limited-range YUV,
        # BT.601). cv2's COLOR_BGR2YUV_I420 emits the same limited-range 4:2:0 signal the
        # PyAV backend feeds NVENC via libswscale (plane values agree to within ~1 level),
        # so both backends encode the same signal. NVENC interprets the "NV12" CPU buffer
        # as limited range (unlike "ARGB", which it treats as full range and which made
        # the output ~3 dB further from the source), so the decoded result stays in the
        # input's video range.
        #
        # The I420->NV12 UV interleave writes into preallocated persistent buffers so the
        # output nv12 is byte-identical to the previous per-frame allocations.
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).ravel()
        nv12 = self._nv12
        nv12[: self._hw] = yuv[: self._hw]
        uv = self._uv
        uv[0::2] = yuv[self._hw:self._hw + self._hw // 4]
        uv[1::2] = yuv[self._hw + self._hw // 4:self._hw + self._hw // 2]
        nv12[self._hw:] = uv
        pp = self._nvc.NV_ENC_PIC_PARAMS()
        pp.inputTimeStamp = self._frame_index
        self._frame_index += 1
        for pkt in self._encoder.Encode(nv12, pp):
            self._muxer.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"], pkt["timestamp"])

    def release(self):
        for pkt in self._encoder.EndEncode():
            self._muxer.MuxVideoPacket(bytes(pkt["data"]), pkt["picture_type"], pkt["timestamp"])
        self._muxer.Finalize()
        # FFmpegMuxer only flushes its interleave buffer to disk in its destructor
        # (av_write_trailer), so it must be explicitly deleted before release() returns.
        del self._muxer
        del self._encoder

class VideoWriter:
    def _parse_encoder_options(self, encoder_options: str):
        tokens = shlex.split(encoder_options)
        parsed_encoder_options = {
            tokens[i].lstrip("-"): tokens[i + 1]
            for i in range(0, len(tokens), 2)
        }
        return parsed_encoder_options

    def __init__(self, output_path, width, height, fps, encoder: str, encoder_options: str, time_base=None, mp4_fast_start=False, gpu_id="0"):
        self._pync_backend = None
        if _is_nvenc_encoder(encoder) and is_pync_encoding_available():
            try:
                self._pync_backend = PyncVideoWriter(
                    output_path, width, height, fps, encoder, encoder_options,
                    time_base=time_base, mp4_fast_start=mp4_fast_start, gpu_id=gpu_id)
                logger.info("Using PyNvVideoCodec backend for encoder '%s'", encoder)
                return
            except Exception:
                logger.warning("PyNvVideoCodec backend failed to start for encoder '%s'; falling back to PyAV", encoder, exc_info=True)
        container_options = {}
        if mp4_fast_start and (output_path.lower().endswith(".mp4") or output_path.lower().endswith(".mov")):
            container_options["movflags"] = "+frag_keyframe+empty_moov+faststart"

        output_container = av.open(output_path, "w", options=container_options)
        video_stream_out: av.VideoStream = output_container.add_stream(encoder, fps)

        self.is_qsv_encoder = 'qsv' in encoder.lower()

        if encoder == "libsvtav1" and "SVT_LOG" not in os.environ:
            # Suppress logging default info messages
            os.environ["SVT_LOG"] = "1"

        target_pix_fmt = 'yuv420p'
        if self.is_qsv_encoder:
            target_pix_fmt = 'nv12'
        
        video_stream_out.pix_fmt = target_pix_fmt
        video_stream_out.codec_context.pix_fmt = target_pix_fmt

        video_stream_out.width = width
        video_stream_out.height = height
        video_stream_out.thread_count = 0
        video_stream_out.thread_type = 3
        video_stream_out.time_base = time_base

        # up until PyAV 15.5.0 it was enough to set these settings on the stream only.
        video_stream_out.codec_context.width = width
        video_stream_out.codec_context.height = height
        video_stream_out.codec_context.thread_count = 0
        video_stream_out.codec_context.thread_type = 3
        video_stream_out.codec_context.time_base = time_base

        stream_options = self._parse_encoder_options(encoder_options)
        # hevc_videotoolbox needs tag 'hvc1' for compatibility (e.g. Safari in MP4/MOV); preset -tag:v is often ignored by PyAV/ffmpeg
        if encoder == 'hevc_videotoolbox':
            stream_options['tag'] = 'hvc1'
        video_stream_out.options = stream_options
        self.output_container = output_container
        self.video_stream = video_stream_out

        # Buffers for reordering frames
        self.BUFFER_MAX_SIZE = 30
        self.pts_heap = []
        self.frame_queue = deque()
        self.pts_set = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def _process_buffer(self, flush_all=False):
        """Processes the buffer to encode frames."""
        if len(self.frame_queue) > (self.BUFFER_MAX_SIZE / 2) or (flush_all and self.frame_queue):
            frame_to_encode = self.frame_queue.popleft()
            pts_to_assign = heapq.heappop(self.pts_heap)
            self.pts_set.remove(pts_to_assign)

            try:
                out_frame = av.VideoFrame.from_numpy_buffer(frame_to_encode, format='bgr24')
            except (ValueError, TypeError):
                # from_numpy_buffer requires a contiguous array; fall back to a copy for exotic inputs
                out_frame = av.VideoFrame.from_ndarray(frame_to_encode, format='bgr24')
            out_frame.pts = pts_to_assign
            out_packet = self.video_stream.encode(out_frame)
            if out_packet:
                self.output_container.mux(out_packet)


    def write(self, frame, frame_pts=None):
        # We add the frame and its pts given by PyAV (FFmpeg) to a FIFO queue and a min heap, respectively.
        # Upon a call to write(), if the buffer is full, we pop the head of the queue and the smallest PTS and pair
        # those together. This operation is a no-op for "nicely behaved" videos, where frames and PTS are decoded
        # in linear order. However, it appears several problematic videos exist such that the frames are given in
        # linear order, but the PTS associated with the frames are not. This strategy is used to avoid prompting
        # the user to identify a framerate ahead of time, and uses the timing of the existing PTS, but reorders the PTS.
        #
        # See https://codeberg.org/ladaapp/lada/pulls/33 for more information/discussion.
        # Frames are expected in BGR (as produced by VideoReader), matching the 'bgr24' format used by _process_buffer.
        if self._pync_backend is not None:
            self._pync_backend.write(frame, frame_pts)
            return
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        if frame_pts not in self.pts_set:
            heapq.heappush(self.pts_heap, frame_pts)
            self.frame_queue.append(frame)
            self.pts_set.add(frame_pts)

        self._process_buffer()

    def release(self):
        if self._pync_backend is not None:
            self._pync_backend.release()
            return
        while len(self.frame_queue) > 0:
            self._process_buffer(flush_all=True)
        # Flush the encoder
        try:
            out_packet = self.video_stream.encode(None)
            if out_packet:
                self.output_container.mux(out_packet)
        except:
            # TODO: For half of my test files flushing QSV encoders fail here with "Application provided invalid, non monotonically increasing dts to muxer in stream"
            # This doesn't happen with libx264 or NVENC encoders. The restored file plays fine so let's ignore it for now.
            if self.is_qsv_encoder:
                logger.warning("Error on flushing QSV encoder. Ignoring...")
            else:
                raise
        self.output_container.close()

def is_video_file(file_path):
    SUPPORTED_VIDEO_FILE_EXTENSIONS = {".asf", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",
                                       ".webm"}

    file_ext = os.path.splitext(file_path)[1]
    return file_ext.lower() in SUPPORTED_VIDEO_FILE_EXTENSIONS

@dataclass
class Encoder:
    name: str
    long_name: str
    hardware_encoder: bool
    hardware_devices: set[str]

    def __hash__(self): return hash(self.name)

def get_human_readable_hardware_device_name(device_type_name: str) -> str:
    if device_type_name == 'qsv':
        return 'Intel QSV'
    elif device_type_name == 'amf':
        return 'AMD AMF'
    elif device_type_name == 'cuda':
        return 'Nvidia CUDA'
    elif device_type_name == 'videotoolbox':
        return 'Apple VideoToolbox'
    return device_type_name

def get_video_encoder_codecs() -> list[Encoder]:
    codecs = set()
    for name in av.codec.codecs_available:
        try:
            codec = av.codec.Codec(name, "w")
        except ValueError:
            continue
        if codec.type != 'video':
            continue
        if re.search(r'\bimage\b', codec.long_name, re.IGNORECASE):
            continue
        codec_long_name = codec.long_name.lower()
        whitelist_video_codecs = ['hevc', 'h265', "h.265", "h264", "h.264", "vp9", "av1", "ffmpeg video codec #1", "huffyuv", "prores", "mpeg-2"]
        whitelist_hardware_devices = ['qsv', 'cuda', 'amf', 'videotoolbox']
        if not any(name in codec_long_name for name in whitelist_video_codecs):
            continue
        is_hardware_encoder = codec.hardware_configs is not None and len(codec.hardware_configs) > 0
        hardware_devices = set([hwconfig.device_type.name for hwconfig in filter(lambda hwconfig: hwconfig.device_type.name in whitelist_hardware_devices, codec.hardware_configs)] if is_hardware_encoder else [] if is_hardware_encoder else [])
        encoder = Encoder(codec.name, codec.long_name, is_hardware_encoder, hardware_devices)
        codecs.add(encoder)
    return sorted(list(codecs), key=lambda e: e.name)

class VideoThumbnailer:

    def __init__(self, video_path: str, thumb_width: int, thumb_height: int):
        self.video_path = video_path
        self.cap = None
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self._frame_cache = {} # LRU cache for recently accessed frames to avoid re-seeking for nearby timestamps
        self._cache_max_size = 60
        self._cache_access_order = []  # Track access order for LRU

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise Exception(f"Unable to open video file: {self.video_path}")

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self._frame_cache.clear()
        self._cache_access_order.clear()

    def _get_fallback_thumbnail(self):
        return np.zeros(shape=(self.thumb_height, self.thumb_width, 3), dtype=np.uint8)

    def _get_cached_thumbnail(self, timestamp_ms: float) -> np.ndarray | None:
        """Get frame from cache if available and recent enough"""
        # Round to nearest 100ms for caching (avoids too many cache entries)
        cache_key = round(timestamp_ms / 100) * 100

        if cache_key in self._frame_cache:
            # Move to end of access order (most recently used)
            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)
            self._cache_access_order.append(cache_key)
            return self._frame_cache[cache_key].copy()
        return None

    def _cache_thumbnail(self, timestamp_ms: float, frame: np.ndarray):
        """Add frame to cache with LRU eviction"""
        cache_key = round(timestamp_ms / 100) * 100

        # Remove from access order if already exists
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)

        # Add to cache
        self._frame_cache[cache_key] = frame.copy()
        self._cache_access_order.append(cache_key)

        # Evict least recently used if cache is full
        if len(self._frame_cache) > self._cache_max_size:
            oldest_key = self._cache_access_order.pop(0)
            del self._frame_cache[oldest_key]

    def get_thumbnail(self, timestamp_ns: int) -> np.ndarray:
        try:
            # Convert nanoseconds to milliseconds for OpenCV
            timestamp_ms = timestamp_ns / 1_000_000

            cached_thumbnail = self._get_cached_thumbnail(timestamp_ms)
            if cached_thumbnail is not None:
                return cached_thumbnail

            self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

            ret, frame = self.cap.read()

            if ret and frame is not None:
                thumbnail = cv2.resize(frame, (self.thumb_width, self.thumb_height), interpolation=cv2.INTER_LINEAR)
                self._cache_thumbnail(timestamp_ms, thumbnail)

                return thumbnail

            return self._get_fallback_thumbnail()

        except Exception as e:
            logger.error(f"Error generating thumbnail at {timestamp_ns}: {e}")
            return self._get_fallback_thumbnail()


class AsyncVideoWriter:
    """Wraps a synchronous :class:`VideoWriter` in a background thread so the
    producer doesn't block on the PyAV/NVENC encode call (mostly CPU-side
    overhead in the FFmpeg/PyNvVideoCodec wrappers).

    Encoding latency is overlapped with the next frames' processing. A bounded
    queue provides backpressure: when the encoder can't keep up, write() blocks
    and the observed throughput converges to the encoder's ceiling. The queue
    must not be unbounded, otherwise the producer would run away from the
    encoder and memory usage would explode.

    Used by both the CLI and the GUI export. The GUI's pause/resume feature is
    compatible: the writer is created once and kept open across a pause; frames
    queued before the pause are flushed in order, and resumed frames queue
    behind them (strict FIFO ordering of timestamps is preserved).
    """

    def __init__(self, output_path, width, height, fps, encoder: str, encoder_options: str, time_base=None, mp4_fast_start=False):
        self._video_writer = VideoWriter(output_path, width, height, fps, encoder=encoder,
                                         encoder_options=encoder_options, time_base=time_base,
                                         mp4_fast_start=mp4_fast_start)
        self._frame_queue: Queue = Queue(maxsize=64)
        self._error: Exception | None = None
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="async-encoder")
        self._writer_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _writer_loop(self):
        try:
            while True:
                item = self._frame_queue.get()
                if item is STOP_MARKER:
                    break
                (frame, frame_pts) = item
                self._video_writer.write(frame, frame_pts)
            self._video_writer.release()
        except Exception as e:
            self._error = e
            logger.error("AsyncVideoWriter writer thread failed: %r", e, exc_info=True)
            try:
                self._video_writer.release()
            except Exception:
                pass

    def write(self, frame, frame_pts=None):
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._frame_queue.put((frame, frame_pts), timeout=0.1)
                return
            except Full:
                logger.debug("AsyncVideoWriter.write: queue full, blocking (pts=%s)", frame_pts)
                continue

    def close(self):
        while True:
            if self._error is not None:
                break
            try:
                self._frame_queue.put(STOP_MARKER, timeout=0.1)
                break
            except Full:
                continue
        self._writer_thread.join()
        if self._error is not None:
            raise self._error

    def release(self):
        self.close()
