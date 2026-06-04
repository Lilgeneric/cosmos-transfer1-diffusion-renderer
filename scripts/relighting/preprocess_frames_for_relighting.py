#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess a pre-cropped image dataset for the Cosmos Diffusion Renderer pipeline.

Steps:
  1. Sort images in input_dir by filename (alphabetical)
  2. Copy to output_dir with sequential zero-padded names (00000.png, 00001.png, ...)
  3. Save original→new mapping to a CSV file for later renaming back

Usage:
    python scripts/relighting/preprocess_frames_for_relighting.py \
        --input_dir  /path/to/wrist_1/cropped/ \
        --output_dir asset/examples/pipeline_input_wrist_1/wrist_1/ \
        --csv_out    wrist_1_rename_mapping.csv
"""

import argparse
import csv
import os
import shutil
import sys


IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')


def main():
    parser = argparse.ArgumentParser(
        description="Sort and sequentially rename pre-cropped images, saving a CSV mapping."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing pre-cropped image files (original filenames)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for sequentially named images (00000.png ...)",
    )
    parser.add_argument(
        "--csv_out", type=str, required=True,
        help="Path to output CSV file (new_name,original_name)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input_dir not found: {args.input_dir}")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not files:
        print(f"ERROR: no images found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)) or ".", exist_ok=True)

    n_digits = max(5, len(str(len(files) - 1)))

    print(f"Input : {args.input_dir}  ({len(files)} images)")
    print(f"Output: {args.output_dir}")
    print(f"CSV   : {args.csv_out}")

    with open(args.csv_out, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["new_name", "original_name"])
        for idx, fname in enumerate(files):
            new_name = f"{idx:0{n_digits}d}.png"
            src = os.path.join(args.input_dir, fname)
            dst = os.path.join(args.output_dir, new_name)
            shutil.copy2(src, dst)
            writer.writerow([new_name, fname])
            if (idx + 1) % 500 == 0 or idx == len(files) - 1:
                print(f"\r  {idx + 1}/{len(files)} frames", end="", flush=True)

    print(f"\nDone. {len(files)} frames → {args.output_dir}")


if __name__ == "__main__":
    main()
