#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extract individual frames from forward renderer output and rename them back
to their original filenames using the rename mapping CSV.

Reads directly from wrist1_forward/relit_frames_XXXX/wrist_1/ JPEGs
(avoids re-extracting from MP4 — no extra compression loss).

Usage:
    python scripts/relighting/restore_relighted_frame_names.py \
        --forward_dir  asset/example_results/wrist_1_forward/ \
        --rename_csv   wrist_1_rename_mapping.csv \
        --output_dir   asset/example_results/wrist_1_renamed/ \
        --num_input_frames 12251 \
        --chunk_size 57 --overlap 8
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict


def load_rename_map(csv_path: str) -> dict:
    """Load CSV into {new_name: original_name} dict."""
    mapping = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["new_name"]] = row["original_name"]
    return mapping


def collect_chunk_frames(clip_dir: str) -> dict:
    """
    Scan clip_dir for {chunk:04d}.{frame:04d}.jpg files.
    Returns {chunk_idx: sorted list of (frame_idx, filepath)}.
    """
    chunks = defaultdict(list)
    for fname in os.listdir(clip_dir):
        parts = fname.split(".")
        if len(parts) != 3 or parts[2].lower() not in ("jpg", "jpeg", "png"):
            continue
        try:
            chunk_idx = int(parts[0])
            frame_idx = int(parts[1])
        except ValueError:
            continue
        chunks[chunk_idx].append((frame_idx, os.path.join(clip_dir, fname)))
    for k in chunks:
        chunks[k].sort(key=lambda x: x[0])
    return chunks


def build_seq_to_filepath(clip_dir: str, total_frames: int,
                           chunk_size: int, overlap: int) -> dict:
    """
    Apply the same overlap/padding removal logic as stitch_gbuffer_video.py
    to build {sequential_index: source_filepath}.
    """
    step = chunk_size - overlap
    chunks = collect_chunk_frames(clip_dir)
    sorted_chunks = sorted(chunks.keys())

    seq_to_path = {}
    frames_written = 0

    for order, chunk_idx in enumerate(sorted_chunks):
        chunk_start = chunk_idx * step
        real_frames = max(0, min(chunk_size, total_frames - chunk_start))
        first_keep = overlap if order > 0 else 0
        last_keep = real_frames

        if first_keep >= last_keep:
            continue

        frame_lookup = {fi: fp for fi, fp in chunks[chunk_idx]}
        for fi in range(first_keep, last_keep):
            if fi in frame_lookup:
                seq_to_path[frames_written] = frame_lookup[fi]
            frames_written += 1

    return seq_to_path


def process_env(env_dir: str, clip_name: str, out_env_dir: str,
                rename_map: dict, total_frames: int,
                chunk_size: int, overlap: int, ext: str):
    clip_dir = os.path.join(env_dir, clip_name)
    if not os.path.isdir(clip_dir):
        print(f"  [SKIP] not found: {clip_dir}")
        return

    seq_to_path = build_seq_to_filepath(clip_dir, total_frames, chunk_size, overlap)
    os.makedirs(out_env_dir, exist_ok=True)

    missing_csv = 0
    missing_src = 0
    copied = 0

    for seq_idx, src_path in sorted(seq_to_path.items()):
        new_name_key = f"{seq_idx:05d}.png"          # key in CSV (always .png)
        original_name = rename_map.get(new_name_key)
        if original_name is None:
            missing_csv += 1
            continue

        # Replace extension with actual source extension
        stem = os.path.splitext(original_name)[0]
        dst_name = f"{stem}{ext}"
        dst_path = os.path.join(out_env_dir, dst_name)

        if not os.path.exists(src_path):
            missing_src += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

        if copied % 1000 == 0:
            print(f"\r    {copied}/{len(seq_to_path)} frames", end="", flush=True)

    print(f"\r    {copied} frames → {out_env_dir}  "
          f"(missing_csv={missing_csv}, missing_src={missing_src})")


def main():
    parser = argparse.ArgumentParser(
        description="Rename relit frames back to original filenames using CSV mapping."
    )
    parser.add_argument(
        "--forward_dir", type=str,
        default="asset/example_results/wrist1_forward/",
        help="Directory containing relit_frames_XXXX/ subdirs",
    )
    parser.add_argument(
        "--rename_csv", type=str,
        default="wrist1_rename_mapping.csv",
        help="CSV with columns new_name,original_name",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="asset/example_results/wrist1_renamed/",
        help="Root output directory; env subdirs created automatically",
    )
    parser.add_argument(
        "--clip_name", type=str, default="wrist_1",
        help="Clip subdirectory name inside each relit_frames_XXXX/",
    )
    parser.add_argument(
        "--env_dirs", type=str, nargs="+",
        default=["relit_frames_0000", "relit_frames_0002",
                 "relit_frames_0003", "relit_frames_0004"],
        help="Which env subdirs to process",
    )
    parser.add_argument("--num_input_frames", type=int, default=12251)
    parser.add_argument("--chunk_size",        type=int, default=57)
    parser.add_argument("--overlap",           type=int, default=8)
    parser.add_argument(
        "--output_ext", type=str, default=".jpg",
        help="Output file extension (.jpg or .png)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.rename_csv):
        print(f"ERROR: CSV not found: {args.rename_csv}")
        sys.exit(1)

    print(f"Loading rename map from {args.rename_csv} ...")
    rename_map = load_rename_map(args.rename_csv)
    print(f"  {len(rename_map)} entries loaded")

    for env_subdir in args.env_dirs:
        env_dir = os.path.join(args.forward_dir, env_subdir)
        out_env_dir = os.path.join(args.output_dir, env_subdir)
        print(f"\nProcessing {env_subdir} ...")
        process_env(
            env_dir=env_dir,
            clip_name=args.clip_name,
            out_env_dir=out_env_dir,
            rename_map=rename_map,
            total_frames=args.num_input_frames,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            ext=args.output_ext,
        )

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
