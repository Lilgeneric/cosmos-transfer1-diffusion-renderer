# Long Video Relighting Pipeline

This document describes the long-video relighting workflow added on top of
Cosmos-Transfer1-DiffusionRenderer. It is written for cropped robot-camera frame
sequences that are too long to process as a single 57-frame clip.

## What This Adds

- Isolated per-clip input directories so one run cannot accidentally process
  other datasets under `asset/examples/`.
- Sequential frame preprocessing with a CSV map from generated names back to the
  original filenames.
- Chunked inverse and forward rendering with `57` frame windows and `8` frame
  overlap.
- A streaming stitcher that writes MP4 files chunk by chunk instead of loading a
  full long video into memory.
- Optional boundary color correction for stitched videos.
- A final frame-rename stage that copies relit JPEG frames back to the original
  source names without re-encoding.
- A terminal monitor for long-running jobs.

## Directory Layout

Tracked project files:

```text
scripts/relighting/
  preprocess_frames_for_relighting.py
  run_long_video_relighting.sh
  monitor_relighting_pipeline.sh
  stitch_chunked_relighting_frames.py
  restore_relighted_frame_names.py

docs/relighting/
  README.md
  execution_plans/
    paper_wrist1_execution_plan.md
```

Generated local files are intentionally ignored by git:

```text
asset/examples/pipeline_input_<clip_name>/
asset/examples/video_frames_examples/
asset/example_results/
<clip_name>_pipeline.log
<clip_name>_rename_mapping.csv
.<clip_name>_pipeline_status
```

The paper-specific relighting execution plan is preserved in
`docs/relighting/execution_plans/paper_wrist1_execution_plan.md`.

## Environment

Install the upstream project dependencies first, then download checkpoints:

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
pip install -r requirements.txt

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) \
python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints
```

The shell pipeline activates `cosmos-predict1` through
`/home/vla/anaconda3/etc/profile.d/conda.sh` by default. On another machine, set
`CONDA_SH` and `CONDA_ENV` before running:

```bash
CONDA_SH=/path/to/conda.sh CONDA_ENV=cosmos-predict1 \
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 1
```

Expected hardware for the default mode:

- NVIDIA GPU with about 27 GB free VRAM.
- Use `--offload` to reduce peak VRAM to roughly 16 GB at the cost of speed.
- Long runs need substantial disk space because inverse G-buffers and relit
  frames are written as images.

## Input Contract

The full pipeline expects a directory of already-cropped image frames:

```text
/path/to/wrist_1/cropped/
  20001.png
  20002.png
  ...
```

Supported extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, and `.tif`.
Files are sorted alphabetically, copied, and renamed to sequential names:

```text
asset/examples/pipeline_input_wrist_1/wrist_1/
  00000.png
  00001.png
  ...
```

The original mapping is stored in:

```text
wrist_1_rename_mapping.csv
```

## Full Pipeline

Run the whole workflow from original cropped frames:

```bash
nohup bash scripts/relighting/run_long_video_relighting.sh \
  --input_dir /path/to/wrist_1/cropped \
  --clip_name wrist_1 \
  >> wrist_1_pipeline.log 2>&1 &
```

Monitor progress from another terminal:

```bash
bash scripts/relighting/monitor_relighting_pipeline.sh wrist_1 10
```

The pipeline runs these stages:

```text
Step 0   preprocess frames and write rename CSV
Step 1   inverse rendering, RGB -> G-buffers
Step 2a  forward rendering for HDRI indices 0, 2, 3
Step 2b  forward rendering for random lighting, env index 4
Step 3   stitch chunked relit frames into MP4 videos
Step 4   copy relit frames back to original filenames
```

Lighting outputs:

```text
relit_frames_0000  sunny_vondelpark_2k.hdr
relit_frames_0002  street_lamp_2k.hdr
relit_frames_0003  rosendal_plains_1_2k.hdr
relit_frames_0004  random lighting
```

## Resume From a Step

Use `--from` to skip completed work:

```bash
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 1
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2b
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 3
bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 4
```

Rules:

- `--from 0` requires `--input_dir`.
- `--from 1` and later only require `--clip_name`, assuming previous outputs
  still exist locally.
- `--from 2b` skips the fixed-HDRI forward pass and only runs random lighting,
  stitching, and frame renaming.

Low-VRAM mode:

```bash
bash scripts/relighting/run_long_video_relighting.sh \
  --clip_name wrist_1 \
  --from 2b \
  --offload
```

## Output Structure

For `--clip_name wrist_1`, generated outputs are:

```text
asset/examples/pipeline_input_wrist_1/
  wrist_1/
    00000.png
    00001.png

asset/example_results/
  wrist_1_inverse/
    gbuffer_frames/
      wrist_1/
        0000.0000.basecolor.jpg
        0000.0000.normal.jpg
        ...

  wrist_1_forward/
    relit_frames_0000/wrist_1/
    relit_frames_0002/wrist_1/
    relit_frames_0003/wrist_1/
    relit_frames_0004/wrist_1/

  wrist_1_final/
    relit_frames_0000/wrist_1.rgb.mp4
    relit_frames_0002/wrist_1.rgb.mp4
    relit_frames_0003/wrist_1.rgb.mp4
    relit_frames_0004/wrist_1.rgb.mp4

  wrist_1_renamed/
    relit_frames_0000/20001.jpg
    relit_frames_0002/20001.jpg
    ...
```

Runtime metadata:

```text
wrist_1_pipeline.log
wrist_1_rename_mapping.csv
.wrist_1_pipeline_status
```

## Manual Stage Commands

Preprocess only:

```bash
python scripts/relighting/preprocess_frames_for_relighting.py \
  --input_dir /path/to/wrist_1/cropped \
  --output_dir asset/examples/pipeline_input_wrist_1/wrist_1 \
  --csv_out wrist_1_rename_mapping.csv
```

Stitch one lighting directory:

```bash
python scripts/relighting/stitch_chunked_relighting_frames.py \
  --gbuffer_frames_dir asset/example_results/wrist_1_forward/relit_frames_0000 \
  --num_input_frames 12251 \
  --output_dir asset/example_results/wrist_1_final/relit_frames_0000 \
  --chunk_size 57 \
  --overlap 8 \
  --fps 24 \
  --gbuffer_passes rgb \
  --color_correct
```

Restore original frame names:

```bash
python scripts/relighting/restore_relighted_frame_names.py \
  --forward_dir asset/example_results/wrist_1_forward \
  --rename_csv wrist_1_rename_mapping.csv \
  --output_dir asset/example_results/wrist_1_renamed \
  --clip_name wrist_1 \
  --num_input_frames 12251 \
  --chunk_size 57 \
  --overlap 8 \
  --output_ext .jpg
```

## Why Chunk Size 57 and Overlap 8

The Cosmos tokenizer uses an `8x8x8` compression pattern. With
`--num_video_frames 57`, the temporal latent length is `57 // 8 + 1 = 8`.
An overlap of `8` frames gives adjacent chunks one full temporal token of shared
context. This is the smallest overlap that meaningfully reduces chunk boundary
flicker while keeping total work manageable.

## Cleanup Policy

Do not commit generated data, logs, rename CSVs, checkpoints, or local input
videos. They are ignored by `.gitignore` and can be regenerated from source
frames and checkpoints.

Keep these files under version control:

- Relighting scripts under `scripts/relighting/`.
- Reproduction docs under `docs/relighting/`.
- The paper execution plan under `docs/relighting/execution_plans/`.
- Small upstream examples such as HDRIs, images, and `asset/teaser.gif`.

## Troubleshooting

- If Step 1 processes more clips than expected, check that `--dataset_path`
  points to `asset/examples/pipeline_input_<clip_name>/`, not the shared
  `asset/examples/` directory.
- If a GPU process keeps VRAM after inference exits, the shell runner starts
  inference in a new process group and force-cleans that group before moving to
  the next stage.
- If monitor output shows zero total frames, Step 0 has not produced
  `asset/examples/pipeline_input_<clip_name>/<clip_name>/*.png` yet.
- If renamed outputs are missing, verify the CSV still exists and its
  `new_name` values match the sequential frame names used during preprocessing.
