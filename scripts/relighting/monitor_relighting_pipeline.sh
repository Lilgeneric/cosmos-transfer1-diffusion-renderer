#!/bin/bash
# Real-time pipeline monitor for relighting pipeline
# Usage: bash scripts/relighting/monitor_relighting_pipeline.sh [clip_name] [refresh_interval_seconds]
#   clip_name defaults to "wrist_1" or reads from status file
# Press Ctrl+C to exit

# ── Resolve clip name ──────────────────────────────────────────────────────────
# Allow: monitor_relighting_pipeline.sh wrist_2 10  OR  monitor_relighting_pipeline.sh 10
if [[ "$1" =~ ^[0-9]+$ ]]; then
    INTERVAL=$1
    CLIP_ARG=""
else
    CLIP_ARG="${1:-}"
    INTERVAL="${2:-10}"
fi

# Try to read clip name from the most recently modified status file
if [ -n "$CLIP_ARG" ]; then
    CLIP_NAME="$CLIP_ARG"
    STATUS_FILE=".${CLIP_NAME}_pipeline_status"
else
    # Auto-detect: find most recent status file
    STATUS_FILE=$(ls -t ./*_pipeline_status ./.??*_pipeline_status 2>/dev/null | head -1)
    if [ -n "$STATUS_FILE" ]; then
        CLIP_NAME=$(grep "^CLIP_NAME=" "$STATUS_FILE" 2>/dev/null | cut -d= -f2)
        CLIP_NAME="${CLIP_NAME:-wrist_1}"
    else
        CLIP_NAME="wrist_1"
        STATUS_FILE=".wrist_1_pipeline_status"
    fi
fi

CHUNK_SIZE=57
OVERLAP=8
STEP_SZ=$(( CHUNK_SIZE - OVERLAP ))

PREPROC_DIR="asset/examples/pipeline_input_${CLIP_NAME}/${CLIP_NAME}"
GBUF_DIR="asset/example_results/${CLIP_NAME}_inverse/gbuffer_frames/${CLIP_NAME}"
FWD_DIR="asset/example_results/${CLIP_NAME}_forward"
FINAL_DIR="asset/example_results/${CLIP_NAME}_final"
RENAMED_DIR="asset/example_results/${CLIP_NAME}_renamed"
LOG="${CLIP_NAME}_pipeline.log"

# Read total frames from preprocessed dir (updated each loop)
get_total_frames() {
    ls "${PREPROC_DIR}"/*.png 2>/dev/null | wc -l
}

# ── Helpers ───────────────────────────────────────────────────────────────────

fmt_duration() {
    local secs=$1
    local d=$(( secs / 86400 ))
    local h=$(( (secs % 86400) / 3600 ))
    local m=$(( (secs % 3600) / 60 ))
    local s=$(( secs % 60 ))
    if [ $d -gt 0 ]; then
        printf "%dd %02dh %02dm %02ds" $d $h $m $s
    elif [ $h -gt 0 ]; then
        printf "%dh %02dm %02ds" $h $m $s
    else
        printf "%dm %02ds" $m $s
    fi
}

bar() {
    local pct=$1
    local width=36
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    printf "\033[32m["
    printf "%${filled}s" | tr ' ' '█'
    printf "\033[90m%${empty}s" | tr ' ' '░'
    printf "\033[32m]\033[0m %3d%%" "$pct"
}

read_status_key() {
    local key=$1
    if [ -f "$STATUS_FILE" ]; then
        grep "^${key}=" "$STATUS_FILE" | tail -1 | cut -d= -f2
    fi
}

count_files() {
    local dir=$1
    if [ -d "$dir" ]; then
        ls "$dir" | wc -l
    else
        echo 0
    fi
}

# ── Main loop ─────────────────────────────────────────────────────────────────

while true; do
    NOW=$(date +%s)
    clear

    TOTAL_FRAMES=$(get_total_frames)
    TOTAL_FRAMES=${TOTAL_FRAMES:-0}
    if [ "$TOTAL_FRAMES" -gt 0 ]; then
        TOTAL_CHUNKS=$(( (TOTAL_FRAMES - OVERLAP + STEP_SZ - 1) / STEP_SZ ))
    else
        TOTAL_CHUNKS=0
    fi
    FILES_PER_INV_CHUNK=$(( CHUNK_SIZE * 5 ))
    FILES_PER_FWD_CHUNK=$CHUNK_SIZE

    # Header — pad to fixed width
    HEADER="  ${CLIP_NAME} Relighting Pipeline  │  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "\033[1;36m╔══════════════════════════════════════════════════════════════════╗"
    printf "║%-66s║\n" "$HEADER"
    echo -e "╚══════════════════════════════════════════════════════════════════╝\033[0m"
    echo ""

    # ── CPU / RAM / Swap ─────────────────────────────────────────────────────
    cpu_idle=$(top -bn1 | grep "Cpu(s)" | awk '{for(i=1;i<=NF;i++) if($i~/id,/) {gsub(/[^0-9.]/,"",$i); print $i}}')
    cpu_use=$(echo "100 - ${cpu_idle:-0}" | bc 2>/dev/null || echo "?")

    mem_info=$(grep -E "^(MemTotal|MemAvailable|SwapTotal|SwapFree):" /proc/meminfo)
    mem_total_kb=$(echo "$mem_info" | awk '/MemTotal/    {print $2}')
    mem_avail_kb=$(echo "$mem_info" | awk '/MemAvailable/{print $2}')
    swp_total_kb=$(echo "$mem_info" | awk '/SwapTotal/   {print $2}')
    swp_free_kb=$( echo "$mem_info" | awk '/SwapFree/    {print $2}')
    mem_used_kb=$(( mem_total_kb - mem_avail_kb ))
    swp_used_kb=$(( swp_total_kb - swp_free_kb ))
    mem_pct_sys=$(( mem_used_kb  * 100 / (mem_total_kb  + 1) ))
    swp_pct=$(( swp_used_kb  * 100 / (swp_total_kb + 1) ))
    mem_used_gb=$(echo "scale=1; $mem_used_kb  / 1048576" | bc)
    mem_total_gb=$(echo "scale=1; $mem_total_kb / 1048576" | bc)
    swp_used_gb=$(echo "scale=1; $swp_used_kb  / 1048576" | bc)
    swp_total_gb=$(echo "scale=1; $swp_total_kb / 1048576" | bc)

    echo -e "  \033[1;33mCPU\033[0m  ${cpu_use}%  │  \033[1;33mRAM\033[0m  ${mem_used_gb}/${mem_total_gb} GB (${mem_pct_sys}%)  │  \033[1;33mSWP\033[0m  ${swp_used_gb}/${swp_total_gb} GB (${swp_pct}%)"
    echo ""

    # ── GPU Info ─────────────────────────────────────────────────────────────
    if command -v nvidia-smi &>/dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_info" ]; then
            gpu_name=$(echo "$gpu_info" | cut -d, -f1 | xargs)
            mem_used=$(echo "$gpu_info" | cut -d, -f2 | xargs)
            mem_total=$(echo "$gpu_info" | cut -d, -f3 | xargs)
            gpu_util=$(echo "$gpu_info" | cut -d, -f4 | xargs)
            mem_pct=$(( mem_used * 100 / mem_total ))
            echo -e "  \033[1;33mGPU\033[0m  $gpu_name  │  VRAM: ${mem_used}/${mem_total} MB (${mem_pct}%)  │  Util: ${gpu_util}%"
            echo ""
        fi
    fi

    # ── Step 0: Preprocess ───────────────────────────────────────────────────
    preproc_running=$(pgrep -f "preprocess_frames_for_relighting" | wc -l)
    if [ "$TOTAL_FRAMES" -ge 1 ]; then
        echo -e "  \033[1;32m[Step 0] Preprocess  ✓ COMPLETE\033[0m  (${TOTAL_FRAMES} frames → ${PREPROC_DIR})"
        step0_start=$(read_status_key STEP0_START)
        step0_end=$(read_status_key STEP0_END)
        if [ -n "$step0_start" ] && [ -n "$step0_end" ]; then
            echo -e "          Duration: $(fmt_duration $(( step0_end - step0_start )))"
        fi
    elif [ "$preproc_running" -gt 0 ]; then
        raw_count=$(ls "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg 2>/dev/null | wc -l)
        done_count=$(ls "${PREPROC_DIR}"/*.png 2>/dev/null | wc -l)
        echo -e "  \033[1;34m[Step 0] Preprocess\033[0m  ⟳ running...  ${done_count} frames processed"
    else
        echo -e "  \033[1;34m[Step 0] Preprocess\033[0m  \033[90m· · · not started · · ·\033[0m"
    fi
    echo ""

    # ── Step 1: Inverse Rendering ─────────────────────────────────────────────
    inv_files=$(count_files "$GBUF_DIR")
    inv_chunks=$(( FILES_PER_INV_CHUNK > 0 ? inv_files / FILES_PER_INV_CHUNK : 0 ))
    [ "$TOTAL_CHUNKS" -gt 0 ] && inv_pct=$(( inv_chunks * 100 / TOTAL_CHUNKS )) || inv_pct=0
    inv_partial=$(( (inv_files % FILES_PER_INV_CHUNK) / 5 ))  # frames in current chunk

    if [ $inv_chunks -ge $TOTAL_CHUNKS ]; then
        echo -e "  \033[1;32m[Step 1] Inverse Rendering  ✓ COMPLETE\033[0m  (${inv_chunks}/${TOTAL_CHUNKS} chunks)"

        step1_start=$(read_status_key STEP1_START)
        step1_end=$(read_status_key STEP1_END)
        if [ -n "$step1_start" ] && [ -n "$step1_end" ]; then
            elapsed=$(( step1_end - step1_start ))
            echo -e "          Duration: $(fmt_duration $elapsed)"
        fi
    else
        echo -e "  \033[1;34m[Step 1] Inverse Rendering\033[0m  ($TOTAL_CHUNKS chunks total)"
        printf "          "
        bar $inv_pct
        printf "  %d/%d chunks\n" $inv_chunks $TOTAL_CHUNKS

        step1_start=$(read_status_key STEP1_START)
        if [ -n "$step1_start" ] && [ $inv_chunks -gt 0 ]; then
            elapsed=$(( NOW - step1_start ))
            rate=$(echo "scale=4; $inv_chunks / $elapsed" | bc 2>/dev/null)  # chunks/sec
            rem=$(( TOTAL_CHUNKS - inv_chunks ))
            eta_sec=$(echo "scale=0; $rem / $rate / 1" | bc 2>/dev/null)
            echo -e "          Elapsed: \033[33m$(fmt_duration $elapsed)\033[0m  │  ETA: \033[33m~$(fmt_duration ${eta_sec:-0})\033[0m  │  Rate: $(echo "scale=1; $inv_chunks * 3600 / $elapsed" | bc 2>/dev/null) chunks/h"
            if [ $inv_partial -gt 0 ]; then
                echo -e "          → In-progress chunk: frame ${inv_partial}/${CHUNK_SIZE}"
            fi
        fi
    fi
    echo ""

    # ── Step 2: Forward Rendering ─────────────────────────────────────────────
    echo -e "  \033[1;34m[Step 2] Forward Rendering\033[0m  ($TOTAL_CHUNKS chunks × 4 lighting conditions)"

    declare -A ENV_NAMES
    ENV_NAMES[0000]="HDRI-0 sunny_vondelpark"
    ENV_NAMES[0002]="HDRI-2 street_lamp     "
    ENV_NAMES[0003]="HDRI-3 rosendal_plains "
    ENV_NAMES[0004]="Random lighting        "

    fwd_any=false
    for ENV in 0000 0002 0003 0004; do
        env_dir="$FWD_DIR/relit_frames_${ENV}/${CLIP_NAME}"
        fwd_files=$(count_files "$env_dir")
        fwd_chunks=$(( fwd_files / FILES_PER_FWD_CHUNK ))
        fwd_partial=$(( fwd_files % FILES_PER_FWD_CHUNK ))

        if [ $fwd_files -gt 0 ]; then
            fwd_any=true
            fwd_pct=$(( fwd_chunks * 100 / TOTAL_CHUNKS ))
            if [ $fwd_chunks -ge $TOTAL_CHUNKS ]; then
                printf "    \033[32m%-24s ✓ COMPLETE  %d/%d\033[0m\n" "${ENV_NAMES[$ENV]}" $fwd_chunks $TOTAL_CHUNKS
            else
                printf "    \033[33m%-24s\033[0m " "${ENV_NAMES[$ENV]}"
                bar $fwd_pct
                printf "  %d/%d\n" $fwd_chunks $TOTAL_CHUNKS
                if [ $fwd_partial -gt 0 ]; then
                    echo -e "                             → partial: ${fwd_partial}/${FILES_PER_FWD_CHUNK} frames"
                fi
            fi
        else
            printf "    \033[90m%-24s  · · · waiting · · ·\033[0m\n" "${ENV_NAMES[$ENV]}"
        fi
    done

    # Step 2 timing
    step2_start=$(read_status_key STEP2_START)
    fwd_total=$(count_files "$FWD_DIR/relit_frames_0000/${CLIP_NAME}")
    fwd_chunks_done=$(( fwd_total / FILES_PER_FWD_CHUNK ))
    if [ "$fwd_any" = true ] && [ -n "$step2_start" ] && [ $fwd_chunks_done -gt 0 ]; then
        elapsed2=$(( NOW - step2_start ))
        rate2=$(echo "scale=4; $fwd_chunks_done / $elapsed2" | bc 2>/dev/null)
        # 2a: TOTAL_CHUNKS for 3 env lights (sequential per-chunk but all in one call)
        # 2b: TOTAL_CHUNKS for 1 env light
        # total forward work ≈ 2 runs of TOTAL_CHUNKS
        total_fwd_chunks=$(( TOTAL_CHUNKS * 2 ))
        done_2a=$(( fwd_chunks_done ))
        done_2b=$(count_files "$FWD_DIR/relit_frames_0004/${CLIP_NAME}")
        done_2b=$(( done_2b / FILES_PER_FWD_CHUNK ))
        total_done=$(( done_2a + done_2b ))
        rem_fwd=$(( total_fwd_chunks - total_done ))
        if [ $rem_fwd -gt 0 ] && [ -n "$rate2" ]; then
            eta2=$(echo "scale=0; $rem_fwd / $rate2 / 1" | bc 2>/dev/null)
            echo -e "    Elapsed: \033[33m$(fmt_duration $elapsed2)\033[0m  │  ETA: \033[33m~$(fmt_duration ${eta2:-0})\033[0m  │  Rate: $(echo "scale=1; $fwd_chunks_done * 3600 / $elapsed2" | bc 2>/dev/null) chunks/h"
        fi
    fi
    echo ""

    # ── Step 3: Stitch ────────────────────────────────────────────────────────
    echo -e "  \033[1;34m[Step 3] Stitching (final videos)\033[0m"
    stitch_any=false
    stitch_running=$(pgrep -f "stitch_chunked_relighting_frames" | wc -l)
    for ENV in 0000 0002 0003 0004; do
        mp4="$FINAL_DIR/relit_frames_${ENV}/${CLIP_NAME}.rgb.mp4"
        tmp="$FINAL_DIR/relit_frames_${ENV}/${CLIP_NAME}.rgb.mp4.tmp.mp4"
        if [ -f "$mp4" ]; then
            stitch_any=true
            sz=$(du -h "$mp4" | cut -f1)
            echo -e "    \033[32m✓  relit_frames_${ENV} → ${CLIP_NAME}.rgb.mp4  (${sz})\033[0m"
        elif [ -f "$tmp" ]; then
            stitch_any=true
            sz=$(du -h "$tmp" | cut -f1)
            echo -e "    \033[33m⟳  relit_frames_${ENV}  writing...  (${sz} so far)\033[0m"
        elif [ "$stitch_running" -gt 0 ]; then
            stitch_any=true
            echo -e "    \033[33m·  relit_frames_${ENV}  queued\033[0m"
        fi
    done
    if [ "$stitch_any" = false ]; then
        echo -e "    \033[90m· · · waiting for Step 2 to complete · · ·\033[0m"
    fi
    echo ""

    # ── Step 4: Rename frames ─────────────────────────────────────────────────
    echo -e "  \033[1;34m[Step 4] Rename frames (original filenames)\033[0m"
    rename_running=$(pgrep -f "restore_relighted_frame_names" | wc -l)
    rename_any=false
    rename_total=0
    rename_done_all=true
    for ENV in 0000 0002 0003 0004; do
        env_out="$RENAMED_DIR/relit_frames_${ENV}"
        n=$(count_files "$env_out")
        rename_total=$(( rename_total + n ))
        if [ "$n" -ge "$TOTAL_FRAMES" ]; then
            rename_any=true
            echo -e "    \033[32m✓  relit_frames_${ENV}  ${n}/${TOTAL_FRAMES} frames\033[0m"
        elif [ "$n" -gt 0 ]; then
            rename_any=true
            rename_done_all=false
            pct=$(( n * 100 / TOTAL_FRAMES ))
            printf "    \033[33m⟳  relit_frames_%-4s  " "${ENV}"
            bar $pct
            printf "  %d/%d\033[0m\n" $n $TOTAL_FRAMES
        elif [ "$rename_running" -gt 0 ]; then
            rename_any=true
            rename_done_all=false
            echo -e "    \033[33m·  relit_frames_${ENV}  queued\033[0m"
        else
            rename_done_all=false
        fi
    done
    if [ "$rename_any" = false ]; then
        echo -e "    \033[90m· · · waiting for Step 3 to complete · · ·\033[0m"
    fi
    echo ""

    # ── Pipeline status / log tail ────────────────────────────────────────────
    pipeline_start=$(read_status_key PIPELINE_START)
    if [ -n "$pipeline_start" ]; then
        total_elapsed=$(( NOW - pipeline_start ))
        echo -e "  \033[90mPipeline running for: \033[33m$(fmt_duration $total_elapsed)\033[0m"
    fi

    if [ -f "$LOG" ]; then
        echo -e "  \033[90mLog tail:\033[0m"
        tail -4 "$LOG" 2>/dev/null | grep -v "^$" | while IFS= read -r line; do
            echo -e "    \033[90m$line\033[0m"
        done
    fi

    echo ""
    echo -e "  \033[90mRefreshing every ${INTERVAL}s — Ctrl+C to exit\033[0m"
    sleep "$INTERVAL"
done
