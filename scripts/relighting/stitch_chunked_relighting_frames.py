# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stitch chunked GBuffer output frames into continuous videos (streaming, low memory).

Processes one chunk at a time — memory usage is O(chunk_size × frame_size) regardless
of total video length (typically < 100 MB for 704×1280 frames).

Usage:
    python scripts/relighting/stitch_chunked_relighting_frames.py \
        --gbuffer_frames_dir asset/example_results/video_delighting/gbuffer_frames/ \
        --num_input_frames 255 \
        --output_dir asset/example_results/video_delighting/stitched/ \
        --chunk_size 57 --overlap 8 --fps 10 --color_correct
"""

import argparse
import os
import sys
from collections import defaultdict

import cv2
import imageio
import numpy as np


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_image(fpath: str) -> np.ndarray:
    """Read image as float32 RGB array in [0, 255]."""
    img = cv2.imread(fpath)
    if img is None:
        raise IOError(f"Could not read image: {fpath}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def open_video_writer(out_path: str, fps: int, height: int, width: int):
    """Open an imageio/ffmpeg streaming writer for H.264 MP4."""
    return imageio.get_writer(
        out_path,
        fps=fps,
        quality=5,
        macro_block_size=1,
        ffmpeg_params=["-s", f"{width}x{height}"],
        output_params=["-f", "mp4"],
    )


# ---------------------------------------------------------------------------
# Color correction
# ---------------------------------------------------------------------------

def color_correct_chunk(prev_tail: list, curr_frames: list, n_ref: int) -> list:
    """
    Adjust curr_frames so its color statistics match the tail of prev_tail.
    Only loads n_ref frames from each side — O(n_ref × frame_size) memory.
    """
    n_ref = min(n_ref, len(prev_tail), len(curr_frames))
    if n_ref == 0:
        return curr_frames

    ref = np.stack(prev_tail[-n_ref:], axis=0)   # (n_ref, H, W, 3)
    src = np.stack(curr_frames[:n_ref], axis=0)   # (n_ref, H, W, 3)

    ref_mean = ref.mean(axis=(0, 1, 2)).reshape(1, 1, 3)
    ref_std  = ref.std(axis=(0, 1, 2)).clip(1e-6).reshape(1, 1, 3)
    src_mean = src.mean(axis=(0, 1, 2)).reshape(1, 1, 3)
    src_std  = src.std(axis=(0, 1, 2)).clip(1e-6).reshape(1, 1, 3)

    corrected = []
    for frame in curr_frames:
        f = (frame - src_mean) / src_std * ref_std + ref_mean
        corrected.append(np.clip(f, 0, 255))
    return corrected


# ---------------------------------------------------------------------------
# Frame collection
# ---------------------------------------------------------------------------

def count_input_frames(input_frames_dir: str, clip_name: str) -> int:
    clip_dir = os.path.join(input_frames_dir, clip_name)
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"Input clip directory not found: {clip_dir}")
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return len([f for f in os.listdir(clip_dir) if f.lower().endswith(exts)])


def collect_chunk_frames(clip_dir: str, gbuffer_pass: str) -> dict:
    """
    Scan clip_dir for frames in two naming conventions:
      - {chunk:04d}.{frame:04d}.{pass}.ext  (inverse renderer, 4-part)
      - {chunk:04d}.{frame:04d}.ext          (forward renderer, 3-part, pass implicit)
    Returns dict: chunk_index -> sorted list of (frame_index, filepath)
    """
    chunks = defaultdict(list)
    for fname in os.listdir(clip_dir):
        parts = fname.split('.')
        if len(parts) == 4:
            if parts[2] != gbuffer_pass or parts[3].lower() not in ('jpg', 'jpeg', 'png'):
                continue
            try:
                chunk_idx = int(parts[0])
                frame_idx = int(parts[1])
            except ValueError:
                continue
        elif len(parts) == 3:
            if parts[2].lower() not in ('jpg', 'jpeg', 'png'):
                continue
            try:
                chunk_idx = int(parts[0])
                frame_idx = int(parts[1])
            except ValueError:
                continue
        else:
            continue
        chunks[chunk_idx].append((frame_idx, os.path.join(clip_dir, fname)))
    for k in chunks:
        chunks[k].sort(key=lambda x: x[0])
    return chunks


# ---------------------------------------------------------------------------
# Core stitch logic  (streaming — O(chunk) memory)
# ---------------------------------------------------------------------------

def stitch_clip(
    clip_dir: str,
    clip_name: str,
    output_dir: str,
    total_input_frames: int,
    chunk_size: int,
    overlap: int,
    fps: int,
    gbuffer_passes: list,
    color_correct: bool,
):
    step = chunk_size - overlap

    for gbuffer_pass in gbuffer_passes:
        chunks = collect_chunk_frames(clip_dir, gbuffer_pass)
        if not chunks:
            continue

        sorted_chunk_indices = sorted(chunks.keys())
        n_chunks = len(sorted_chunk_indices)

        out_path     = os.path.join(output_dir, f"{clip_name}.{gbuffer_pass}.mp4")
        out_path_tmp = out_path + ".tmp.mp4"

        writer       = None
        prev_tail    = []   # last `overlap` frames of the previous chunk (float32 RGB)
        frames_written = 0

        for order, chunk_idx in enumerate(sorted_chunk_indices):
            chunk_start = chunk_idx * step
            real_frames = max(0, min(chunk_size, total_input_frames - chunk_start))

            first_keep = overlap if order > 0 else 0
            last_keep  = real_frames

            if first_keep >= last_keep:
                continue

            frame_lookup = {fi: fp for fi, fp in chunks[chunk_idx]}
            chunk_frames = []
            for fi in range(first_keep, last_keep):
                if fi in frame_lookup:
                    chunk_frames.append(read_image(frame_lookup[fi]))
                else:
                    print(f"\n  WARNING: missing frame {fi} in chunk {chunk_idx} [{gbuffer_pass}]")

            if not chunk_frames:
                continue

            # Open writer on first batch (we now know H, W)
            if writer is None:
                h, w = chunk_frames[0].shape[:2]
                writer = open_video_writer(out_path_tmp, fps, h, w)

            # Color correct at chunk boundary
            if color_correct and order > 0 and prev_tail:
                chunk_frames = color_correct_chunk(prev_tail, chunk_frames, n_ref=overlap)

            # Stream frames to disk immediately
            for f in chunk_frames:
                writer.append_data(np.clip(f, 0, 255).astype(np.uint8))
                frames_written += 1

            # Keep only boundary tail for next chunk's color correction
            prev_tail = chunk_frames[-overlap:]

            print(
                f"\r  [{gbuffer_pass}] chunk {order+1}/{n_chunks} | {frames_written}/{total_input_frames} frames",
                end="", flush=True,
            )

        print()  # newline after progress

        if writer is not None:
            writer.close()
            os.replace(out_path_tmp, out_path)   # atomic rename → monitor sees mp4 only when complete
            print(f"  [{gbuffer_pass}] {frames_written} frames → {out_path}")
        else:
            print(f"  [SKIP] No frames for {clip_name}.{gbuffer_pass}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stitch chunked GBuffer frames into continuous H.264 videos (streaming)."
    )
    parser.add_argument(
        "--gbuffer_frames_dir", type=str, required=True,
        help="Directory containing per-clip chunk frame subdirectories",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for stitched .mp4 files",
    )
    parser.add_argument(
        "--input_frames_dir", type=str, default=None,
        help="Original input frames dir; used to count total frames per clip",
    )
    parser.add_argument(
        "--num_input_frames", type=int, default=None,
        help="Total input frame count (overrides --input_frames_dir)",
    )
    parser.add_argument("--chunk_size", type=int, default=57)
    parser.add_argument("--overlap",    type=int, default=8)
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument(
        "--gbuffer_passes", type=str, nargs="+",
        default=["basecolor", "normal", "depth", "roughness", "metallic"],
    )
    parser.add_argument(
        "--color_correct", action="store_true",
        help="Apply histogram matching at chunk boundaries to reduce color drift",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clip_names = sorted(
        d for d in os.listdir(args.gbuffer_frames_dir)
        if os.path.isdir(os.path.join(args.gbuffer_frames_dir, d))
    )
    if not clip_names:
        print(f"No clip directories found in {args.gbuffer_frames_dir}")
        sys.exit(1)

    print(f"Found {len(clip_names)} clip(s): {clip_names}")
    if args.color_correct:
        print("Color correction: ENABLED (histogram matching at chunk boundaries)")

    for clip_name in clip_names:
        clip_dir = os.path.join(args.gbuffer_frames_dir, clip_name)
        print(f"\nProcessing clip: {clip_name}")

        if args.num_input_frames is not None:
            total_frames = args.num_input_frames
        elif args.input_frames_dir is not None:
            try:
                total_frames = count_input_frames(args.input_frames_dir, clip_name)
            except FileNotFoundError as e:
                print(f"  WARNING: {e}. Using heuristic.")
                total_frames = None
        else:
            total_frames = None

        if total_frames is None:
            probe = collect_chunk_frames(clip_dir, args.gbuffer_passes[0])
            n = len(probe)
            step = args.chunk_size - args.overlap
            total_frames = step * (n - 1) + args.chunk_size
            print(f"  Estimated total_frames = {total_frames} (from {n} chunks)")
        else:
            print(f"  Total input frames = {total_frames}")

        stitch_clip(
            clip_dir=clip_dir,
            clip_name=clip_name,
            output_dir=args.output_dir,
            total_input_frames=total_frames,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            fps=args.fps,
            gbuffer_passes=args.gbuffer_passes,
            color_correct=args.color_correct,
        )

    print(f"\nDone. Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
