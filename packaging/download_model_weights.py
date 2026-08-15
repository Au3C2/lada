#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Download the full Lada model weight set and verify its checksums.

The manifest model_weights/checksums_sha256.txt is the single source of truth
for which weights exist and what their sha256 must be. Every manifest entry
with a known download URL is fetched into model_weights/; files that are
already present and valid are skipped, so re-running is cheap. The script
exits non-zero if any download cannot be verified, which makes it safe to use
in CI and inside the Docker build.

Entries without a public download URL (currently
lada_mosaic_restoration_model_bj_pov.pth) are reported and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_BASE_URL = "https://huggingface.co/ladaapp/lada/resolve/main/{}?download=true"
USER_AGENT = "lada-release-builder"

# Files whose download URL is not the Lada Hugging Face repository.
EXTRA_URLS = {
    "3rd_party/clean_youknow_video.pth": (
        "https://drive.usercontent.google.com/download"
        "?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t"
    ),
    # The official GitHub release asset (notAI-tech/NudeNet v3.4-weights) now serves
    # rate-limit pages to anonymous downloads; this HF mirror is byte-identical to it
    # (sha256 verified against the manifest).
    "3rd_party/640m.pt": "https://huggingface.co/vladmandic/nudenet/resolve/main/nudenet-v34-640m.pt",
    "3rd_party/DOVER.pth": "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
    "3rd_party/spynet_20210409-c6c1bd09.pth": "https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth",
    "3rd_party/vgg19-dcbb9e9d.pth": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "3rd_party/centerface.onnx": "https://github.com/ORB-HD/deface/raw/refs/tags/v1.5.0/deface/centerface.onnx",
    "3rd_party/ch_head_s_1536_e150_best_mMR.pt": "https://huggingface.co/HoyerChou/BPJDet/resolve/main/ch_head_s_1536_e150_best_mMR.pt?download=true",
}

# Weights listed in the manifest but with no public download URL.
NO_URL = {"lada_mosaic_restoration_model_bj_pov.pth"}

RETRIES = 3


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(weights_dir: Path) -> dict[str, str]:
    manifest_path = weights_dir / "checksums_sha256.txt"
    manifest: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            manifest[parts[1]] = parts[0]
    return manifest


def resolve_url(relative_path: str) -> str | None:
    if relative_path in EXTRA_URLS:
        return EXTRA_URLS[relative_path]
    if not relative_path.startswith("3rd_party/"):
        return HF_BASE_URL.format(relative_path)
    return None


def download_to_temp(url: str, tmp_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        with open(tmp_path, "wb") as fh:
            while chunk := response.read(1024 * 1024):
                fh.write(chunk)


def download_verified(url: str, dest_path: Path, expected_sha256: str) -> None:
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            download_to_temp(url, tmp_path)
            actual_sha256 = sha256_of(tmp_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"checksum mismatch: expected {expected_sha256}, got {actual_sha256}")
            os.replace(tmp_path, dest_path)
            return
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise
            print(f"  retry {attempt}/{RETRIES} for {dest_path.name}: {exc}", file=sys.stderr)
            time.sleep(5 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "model_weights",
        help="Directory to download weights into (default: <project>/model_weights)",
    )
    args = parser.parse_args()

    weights_dir: Path = args.weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(weights_dir)

    failures = []
    skipped = []
    for relative_path, expected_sha256 in manifest.items():
        if relative_path in NO_URL:
            print(f"SKIP {relative_path}: no public download URL")
            skipped.append(relative_path)
            continue
        url = resolve_url(relative_path)
        if url is None:
            print(f"SKIP {relative_path}: no download URL configured")
            skipped.append(relative_path)
            continue

        dest_path = weights_dir / relative_path
        if dest_path.is_file() and sha256_of(dest_path) == expected_sha256:
            print(f"OK   {relative_path}: already present and valid")
            continue

        try:
            print(f"GET  {relative_path}")
            download_verified(url, dest_path, expected_sha256)
            print(f"OK   {relative_path}: {dest_path.stat().st_size / 1024 / 1024:.1f} MiB")
        except Exception as exc:
            print(f"FAIL {relative_path}: {exc}", file=sys.stderr)
            failures.append(relative_path)

    if skipped:
        print(f"Skipped {len(skipped)} weight(s) without a download URL: {', '.join(skipped)}")
    if failures:
        print(f"Failed to download {len(failures)} weight(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
