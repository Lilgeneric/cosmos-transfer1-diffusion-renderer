#!/bin/bash
# Full relighting pipeline: pre-cropped dataset → relit frames with original filenames
# Lighting: HDRI-0 (sunny), HDRI-2 (street_lamp), HDRI-3 (rosendal), random
#
# Usage:
#   bash scripts/relighting/run_long_video_relighting.sh --input_dir /path/to/wrist_1/cropped/ --clip_name wrist_1
#   bash scripts/relighting/run_long_video_relighting.sh --input_dir /path/to/wrist_2/cropped/ --clip_name wrist_2
#   bash scripts/relighting/run_long_video_relighting.sh --input_dir /path/to/wrist_3/cropped/ --clip_name wrist_3
#
# Skip steps (e.g. preprocessing already done):
#   bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 1    # skip preprocess
#   bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2    # skip inverse
#   bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2b   # skip Step 2a (HDRI 0,2,3), only run random+stitch+rename
#   bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 3    # skip all forward, only stitch+rename
#   bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 4    # only rename
#
# Steps:
#   0   Preprocess  sort + sequential rename + CSV mapping
#   1   Inverse     RGB → G-buffers (chunked, overlap=8)
#   2a  Forward     G-buffers + HDRI 0,2,3 → relit RGB
#   2b  Forward     G-buffers + random HDRI → relit RGB
#   3   Stitch      chunked frames → continuous MP4 videos
#   4   Rename      sequential frames → original filenames
#
# Low-VRAM mode (reduces peak from ~27 GB to ~16 GB, slower):
#   bash scripts/relighting/run_long_video_relighting.sh ... --offload
#
# Run in background with logging:
#   nohup bash scripts/relighting/run_long_video_relighting.sh \
#       --input_dir /path/to/wrist_1/cropped/ --clip_name wrist_1 \
#       >> wrist_1_pipeline.log 2>&1 &

set -e

# ── Parse arguments ────────────────────────────────────────────────────────────
INPUT_DIR=""
CLIP_NAME=""
FROM_STEP=0
FROM_SUBSTEP=""   # "b" when --from 2b
OFFLOAD_FLAGS=""  # set to "--offload_diffusion_transformer --offload_tokenizer" if --offload

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)  INPUT_DIR="${2%/}"; shift 2 ;;
        --clip_name)  CLIP_NAME="$2";    shift 2 ;;
        --from)
            val="$2"
            if [[ "$val" == "2b" ]]; then
                FROM_STEP=2
                FROM_SUBSTEP=b
            else
                FROM_STEP="$val"
                FROM_SUBSTEP=""
            fi
            shift 2 ;;
        --offload)
            OFFLOAD_FLAGS="--offload_diffusion_transformer --offload_tokenizer"
            shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate: need at least one of --input_dir or --clip_name
if [ -z "$INPUT_DIR" ] && [ -z "$CLIP_NAME" ]; then
    echo "ERROR: --clip_name or --input_dir is required"
    echo ""
    echo "Usage examples:"
    echo "  # Full pipeline from scratch:"
    echo "  bash scripts/relighting/run_long_video_relighting.sh --input_dir /path/to/wrist_1/cropped/ --clip_name wrist_1"
    echo ""
    echo "  # Resume from Step 1 onwards (--input_dir not needed):"
    echo "  bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 1"
    echo ""
    echo "  # Resume from Step 2b only (Step 2a already done):"
    echo "  bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2b"
    echo ""
    echo "  # Low VRAM mode (~16 GB peak instead of ~27 GB):"
    echo "  bash scripts/relighting/run_long_video_relighting.sh --clip_name wrist_1 --from 2b --offload"
    exit 1
fi

# Step 0 requires --input_dir
if [ "$FROM_STEP" -le 0 ] && [ -z "$INPUT_DIR" ]; then
    echo "ERROR: --input_dir is required when running from Step 0"
    echo "  Or skip preprocessing with --from 1"
    exit 1
fi

# Derive clip name from input_dir if not given
if [ -z "$CLIP_NAME" ] && [ -n "$INPUT_DIR" ]; then
    PARENT=$(basename "$(dirname "$INPUT_DIR")")
    if [ "$PARENT" != "." ] && [ "$PARENT" != "/" ]; then
        CLIP_NAME="$PARENT"
    else
        CLIP_NAME=$(basename "$INPUT_DIR")
    fi
fi

# ── Derived paths (all based on CLIP_NAME) ────────────────────────────────────
# Each clip gets its own isolated parent dir to prevent Step 1 from
# accidentally processing other clips sharing the same pipeline_input/ dir.
PREPROC_PARENT="asset/examples/pipeline_input_${CLIP_NAME}"
PREPROC_DIR="${PREPROC_PARENT}/${CLIP_NAME}"
RENAME_CSV="${CLIP_NAME}_rename_mapping.csv"
INVERSE_DIR="asset/example_results/${CLIP_NAME}_inverse"
FORWARD_DIR="asset/example_results/${CLIP_NAME}_forward"
FINAL_DIR="asset/example_results/${CLIP_NAME}_final"
RENAMED_DIR="asset/example_results/${CLIP_NAME}_renamed"
STATUS=".${CLIP_NAME}_pipeline_status"

# ── Environment ────────────────────────────────────────────────────────────────
CONDA_SH="${CONDA_SH:-/home/vla/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cosmos-predict1}"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda init script not found: $CONDA_SH"
    echo "Set CONDA_SH=/path/to/conda.sh or activate the environment manually."
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)

stamp() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]"; }
ts()    { date +%s; }

# ── VRAM-safe inference wrapper ────────────────────────────────────────────────
# Runs Python inference in a new process group (via setsid) so that after the
# main script finishes, any stuck CUDA driver threads can be force-killed as a
# group, freeing VRAM before the next step starts.
#
# Usage: run_inference <python-script-and-args...>
# (implicitly prepends "python")
VRAM_WAIT_SECS=60

run_inference() {
    # setsid creates a new session → process becomes its own group leader
    # PGID == PID of the launched python process
    setsid python "$@" &
    local pid=$!
    local pgid=$pid   # setsid guarantees PGID == PID

    # Wait for inference to finish
    wait "$pid"
    local ec=$?

    if [ $ec -ne 0 ]; then
        echo "$(stamp) ERROR: inference exited with code $ec"
        kill -9 -"$pgid" 2>/dev/null || true
        return $ec
    fi

    # Poll for VRAM to be released (normal cleanup takes a few seconds)
    echo "$(stamp) Inference done — waiting for CUDA context cleanup (up to ${VRAM_WAIT_SECS}s) ..."
    local waited=0
    while [ $waited -lt $VRAM_WAIT_SECS ]; do
        local n
        n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$n" -eq 0 ]; then
            echo "$(stamp) VRAM freed cleanly (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done

    # Still holding VRAM — send SIGKILL to the entire process group
    # This covers the main (zombie) process AND any stuck CUDA driver threads
    echo "$(stamp) WARNING: VRAM still held after ${VRAM_WAIT_SECS}s — force-killing process group $pgid ..."
    kill -9 -"$pgid" 2>/dev/null || true
    sleep 10

    # Last resort: kill threads of any processes still showing in nvidia-smi
    local stuck
    stuck=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {print $1}')
    if [ -n "$stuck" ]; then
        echo "$(stamp) Killing individual threads of stuck PID(s): $stuck"
        for sp in $stuck; do
            for tid in $(ls /proc/"$sp"/task/ 2>/dev/null); do
                kill -9 "$tid" 2>/dev/null || true
            done
            kill -9 "$sp" 2>/dev/null || true
        done
        sleep 10
    fi

    echo "$(stamp) VRAM cleanup forced; continuing."
    return 0
}

echo "$(stamp) Pipeline start"
echo "  clip_name  : $CLIP_NAME"
echo "  input_dir  : $INPUT_DIR"
echo "  from_step  : $FROM_STEP"

# Initialize status file
if [ ! -f "$STATUS" ] || [ "$FROM_STEP" -le 0 ]; then
    { echo "PIPELINE_START=$(ts)"; echo "CLIP_NAME=${CLIP_NAME}"; } > "$STATUS"
else
    grep -v "^PIPELINE_START=" "$STATUS" > "${STATUS}.tmp" 2>/dev/null || true
    echo "PIPELINE_START=$(ts)" > "$STATUS"
    cat "${STATUS}.tmp" >> "$STATUS" 2>/dev/null || true
    rm -f "${STATUS}.tmp"
fi

# ── Step 0: Preprocess ────────────────────────────────────────────────────────
if [ "$FROM_STEP" -le 0 ]; then
    echo "$(stamp) === Step 0: Preprocess (sequential rename + CSV) ==="
    echo "STEP0_START=$(ts)" >> "$STATUS"
    echo "CURRENT_STEP=0" >> "$STATUS"

    python scripts/relighting/preprocess_frames_for_relighting.py \
        --input_dir  "$INPUT_DIR" \
        --output_dir "$PREPROC_DIR" \
        --csv_out    "$RENAME_CSV"

    echo "STEP0_END=$(ts)" >> "$STATUS"
    echo "$(stamp) === Step 0 Done ==="
fi

TOTAL_FRAMES=$(find "${PREPROC_DIR}" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)
echo "  preproc_dir: $PREPROC_DIR"
if [ "$TOTAL_FRAMES" -eq 0 ]; then
    echo "ERROR: No preprocessed frames found in ${PREPROC_DIR}"
    exit 1
fi
echo "$(stamp) Total frames: $TOTAL_FRAMES"

# ── Step 1: Inverse Rendering ─────────────────────────────────────────────────
if [ "$FROM_STEP" -le 1 ]; then
    echo "$(stamp) === Step 1: Inverse Rendering ==="
    echo "STEP1_START=$(ts)" >> "$STATUS"
    echo "CURRENT_STEP=1" >> "$STATUS"

    run_inference cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
        --checkpoint_dir checkpoints \
        --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
        --dataset_path="${PREPROC_PARENT}" \
        --num_video_frames 57 \
        --group_mode folder \
        --chunk_mode all \
        --overlap_n_frames 8 \
        --height 704 --width 1280 \
        --video_save_folder="${INVERSE_DIR}/" \
        --save_image=True \
        --save_video=False \
        ${OFFLOAD_FLAGS}

    echo "STEP1_END=$(ts)" >> "$STATUS"
    echo "$(stamp) === Step 1 Done ==="
fi

# ── Step 2a: Forward Rendering - HDRI 0, 2, 3 ────────────────────────────────
if [ "$FROM_STEP" -le 2 ] && [ "${FROM_SUBSTEP}" != "b" ]; then
    echo "$(stamp) === Step 2a: Forward Rendering HDRI 0, 2, 3 ==="
    echo "STEP2_START=$(ts)" >> "$STATUS"
    echo "CURRENT_STEP=2a" >> "$STATUS"

    run_inference cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
        --checkpoint_dir checkpoints \
        --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
        --dataset_path="${INVERSE_DIR}/gbuffer_frames" \
        --num_video_frames 57 \
        --height 704 --width 1280 \
        --envlight_ind 0 2 3 \
        --use_custom_envmap=True \
        --save_image=True \
        --video_save_folder="${FORWARD_DIR}/" \
        ${OFFLOAD_FLAGS}

    echo "STEP2A_END=$(ts)" >> "$STATUS"
    echo "$(stamp) === Step 2a Done ==="
fi

# ── Step 2b: Forward Rendering - Random lighting ──────────────────────────────
if [ "$FROM_STEP" -le 2 ]; then
    echo "$(stamp) === Step 2b: Forward Rendering Random lighting ==="
    echo "STEP2B_START=$(ts)" >> "$STATUS"
    echo "CURRENT_STEP=2b" >> "$STATUS"

    run_inference cosmos_predict1/diffusion/inference/inference_forward_renderer.py \
        --checkpoint_dir checkpoints \
        --diffusion_transformer_dir Diffusion_Renderer_Forward_Cosmos_7B \
        --dataset_path="${INVERSE_DIR}/gbuffer_frames" \
        --num_video_frames 57 \
        --height 704 --width 1280 \
        --envlight_ind 4 \
        --use_custom_envmap=False \
        --save_image=True \
        --video_save_folder="${FORWARD_DIR}/" \
        ${OFFLOAD_FLAGS}

    echo "STEP2B_END=$(ts)" >> "$STATUS"
    echo "$(stamp) === Step 2b Done ==="
fi

# ── Step 3: Stitch all 4 lighting conditions ──────────────────────────────────
if [ "$FROM_STEP" -le 3 ]; then
    echo "$(stamp) === Step 3: Stitching ==="
    echo "STEP3_START=$(ts)" >> "$STATUS"
    echo "CURRENT_STEP=3" >> "$STATUS"

    for ENVDIR in relit_frames_0000 relit_frames_0002 relit_frames_0003 relit_frames_0004; do
        echo "$(stamp) Stitching $ENVDIR ..."
        mkdir -p "${FINAL_DIR}/${ENVDIR}/"
        python scripts/relighting/stitch_chunked_relighting_frames.py \
            --gbuffer_frames_dir "${FORWARD_DIR}/${ENVDIR}/" \
            --num_input_frames "$TOTAL_FRAMES" \
            --output_dir "${FINAL_DIR}/${ENVDIR}/" \
            --chunk_size 57 --overlap 8 --fps 24 \
            --gbuffer_passes rgb \
            --color_correct
    done

    echo "STEP3_END=$(ts)" >> "$STATUS"
    echo "$(stamp) === Step 3 Done ==="
fi

# ── Step 4: Rename frames back to original filenames ─────────────────────────
echo "$(stamp) === Step 4: Renaming frames ==="
echo "STEP4_START=$(ts)" >> "$STATUS"
echo "CURRENT_STEP=4" >> "$STATUS"

python scripts/relighting/restore_relighted_frame_names.py \
    --forward_dir  "${FORWARD_DIR}/" \
    --rename_csv   "$RENAME_CSV" \
    --output_dir   "${RENAMED_DIR}/" \
    --clip_name    "$CLIP_NAME" \
    --num_input_frames "$TOTAL_FRAMES" \
    --chunk_size 57 --overlap 8 \
    --output_ext .jpg

echo "STEP4_END=$(ts)" >> "$STATUS"
echo "CURRENT_STEP=DONE" >> "$STATUS"
echo "$(stamp) === Step 4 Done ==="

echo ""
echo "$(stamp) === ALL DONE  [${CLIP_NAME}] ==="
echo "Final videos:"
ls -lh "${FINAL_DIR}/relit_frames_"*/
echo "Renamed frames:"
for ENV in relit_frames_0000 relit_frames_0002 relit_frames_0003 relit_frames_0004; do
    n=$(ls "${RENAMED_DIR}/${ENV}/" 2>/dev/null | wc -l)
    echo "  ${ENV}: ${n} frames"
done
