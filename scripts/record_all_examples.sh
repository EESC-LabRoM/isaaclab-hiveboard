#!/usr/bin/env bash
# Record a validation video for every HiveBoard example environment.
#
# Usage:
#   ./scripts/record_all_examples.sh
#   ./scripts/record_all_examples.sh --device cuda:0
#   TASKS="Isaac-HiveBoard-Spot-BallValve-v0" ./scripts/record_all_examples.sh
#   FAST=0 ./scripts/record_all_examples.sh
#
# Extra arguments are forwarded to scripts/play.py.
# Each clip is a single demo: recording stops on success or timeout.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-videos/examples}"
DEVICE="${DEVICE:-cuda:0}"
FAST="${FAST:-1}"
HEADLESS="${HEADLESS:-1}"
FPS="${FPS:-30}"

DEFAULT_TASKS=(
    Isaac-HiveBoard-Spot-BallValve-v0
    Isaac-HiveBoard-Spot-BallValve-Play-v0
    Isaac-HiveBoard-Spot-Button-v0
    Isaac-HiveBoard-Spot-CircuitBreaker-v0
    Isaac-HiveBoard-Spot-HighTorqueValve-v0
    Isaac-HiveBoard-Spot-SmallValve-v0
    Isaac-HiveBoard-Franka-LeverValve-v0
    Isaac-HiveBoard-Franka-CircuitBreaker-v0
)

if [[ -n "${TASKS:-}" ]]; then
    # shellcheck disable=SC2206
    TASK_LIST=(${TASKS})
else
    TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

PLAY_ARGS=(
    --video
    --video_folder "$OUT_DIR"
    --video_fps "$FPS"
    --device "$DEVICE"
    --num_envs 1
)
if [[ "$HEADLESS" == "1" ]]; then
    PLAY_ARGS+=(--headless --enable_cameras)
fi
if [[ "$FAST" == "1" ]]; then
    PLAY_ARGS+=(--fast)
fi

passed=()
failed=()

echo "Recording ${#TASK_LIST[@]} HiveBoard examples to ${OUT_DIR}/"
echo "  one demo per clip  fps=${FPS}  device=${DEVICE}"
echo

for task in "${TASK_LIST[@]}"; do
    slug="${task#Isaac-HiveBoard-}"
    slug="${slug%-v0}"
    slug="${slug//-/_}"
    log_file="$LOG_DIR/${slug}.log"
    echo "=== ${task} ==="

    if uv run python scripts/play.py \
        --task "$task" \
        --video_name "$slug" \
        "${PLAY_ARGS[@]}" \
        "$@" \
        >"$log_file" 2>&1
    then
        video="$(find "$OUT_DIR" -maxdepth 1 -type f -name "${slug}*.mp4" -printf '%T@ %p\n' \
            | sort -nr | head -n1 | cut -d' ' -f2-)"
        if [[ -n "$video" ]]; then
            echo "  ok  ${video}"
            passed+=("$task")
        else
            echo "  fail  play.py exited 0 but no ${slug}*.mp4 in ${OUT_DIR}"
            failed+=("$task")
        fi
    else
        status=$?
        echo "  fail  exit ${status}  (see ${log_file})"
        failed+=("$task")
    fi
    echo
done

echo "Passed (${#passed[@]}):"
if [[ ${#passed[@]} -eq 0 ]]; then
    echo "  (none)"
else
    printf '  %s\n' "${passed[@]}"
fi
echo
echo "Failed (${#failed[@]}):"
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "  (none)"
else
    printf '  %s\n' "${failed[@]}"
fi
echo
echo "Videos: ${OUT_DIR}"
echo "Logs:   ${LOG_DIR}"

if [[ ${#failed[@]} -ne 0 ]]; then
    exit 1
fi
