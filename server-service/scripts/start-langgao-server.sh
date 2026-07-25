#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTOLABEL_ROOT:-/data/mfl/autolabel}"
SERVER="${AUTOLABEL_SERVER_DIR:-$ROOT/server}"
TOKEN_FILE="$ROOT/secrets/api_key"
MODEL_UPLOAD_TOKEN_FILE="$ROOT/secrets/model_upload_key"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; GPU-only startup aborted." >&2
    exit 1
fi

if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "API key file is missing: $TOKEN_FILE" >&2
    exit 1
fi

MAX_WORKERS="${AUTOLABEL_MAX_WORKERS:-4}"
MIN_FREE_MB="${AUTOLABEL_MIN_FREE_MB:-8192}"
MAX_UTIL="${AUTOLABEL_MAX_UTIL:-50}"

if ! [[ "$MAX_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "AUTOLABEL_MAX_WORKERS must be an integer." >&2
    exit 1
fi
if (( MAX_WORKERS < 1 )); then
    MAX_WORKERS=1
elif (( MAX_WORKERS > 4 )); then
    MAX_WORKERS=4
fi

ELIGIBLE_GPU_COUNT="$(
    nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
    awk -F, \
        -v min_free="$MIN_FREE_MB" \
        -v max_util="$MAX_UTIL" \
        -v allowed="${AUTOLABEL_GPU_ID:-}" '
        function is_allowed(id, values, count, i) {
            if (allowed == "") {
                return 1;
            }
            count = split(allowed, values, ",");
            for (i = 1; i <= count; i++) {
                gsub(/ /, "", values[i]);
                if (values[i] == id) {
                    return 1;
                }
            }
            return 0;
        }
        {
            gsub(/ /, "", $1);
            gsub(/ /, "", $2);
            gsub(/ /, "", $3);
            if (is_allowed($1) && ($2 + 0) >= min_free &&
                ($3 + 0) < max_util) {
                eligible += 1;
            }
        }
        END {
            print eligible + 0;
        }
    '
)"

if (( ELIGIBLE_GPU_COUNT < 1 )); then
    EFFECTIVE_WORKERS=1
    echo "No GPU meets the preferred thresholds; starting one fallback worker." >&2
else
    EFFECTIVE_WORKERS="$MAX_WORKERS"
fi

unset CUDA_VISIBLE_DEVICES
export AUTOLABEL_EFFECTIVE_WORKERS="$EFFECTIVE_WORKERS"
export AUTOLABEL_MIN_FREE_MB="$MIN_FREE_MB"
export AUTOLABEL_MAX_UTIL="$MAX_UTIL"
export HOME="$ROOT"
export LANGGAO_GPU_ONLY=1
export X_ANYLABELING_DEVICE=GPU
export XANYLABELING_MODEL_HUB=modelscope
export XANYLABELING_SERVER_CONFIG="$SERVER/configs/langgao-server.yaml"
export XANYLABELING_MODELS_CONFIG="$SERVER/configs/langgao-models.yaml"
export XANYLABELING_API_KEY
XANYLABELING_API_KEY="$(tr -d '\r\n' < "$TOKEN_FILE")"
if [[ -s "$MODEL_UPLOAD_TOKEN_FILE" ]]; then
    export AUTOLABEL_MODEL_UPLOAD_TOKEN
    AUTOLABEL_MODEL_UPLOAD_TOKEN="$(
        tr -d '\r\n' < "$MODEL_UPLOAD_TOKEN_FILE"
    )"
fi
export QT_QPA_PLATFORM=offscreen
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SERVER:$ROOT/src-official-v4.0.0-beta.13:$ROOT/runtime/python-gpu-site-packages:$ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting Langgao AutoLabel server with $EFFECTIVE_WORKERS worker(s)" >&2
echo "GPU policy: 60% utilization + 40% memory pressure, with soft per-worker reservations" >&2
echo "Listening on 127.0.0.1:18618 (SSH tunnel required)" >&2

cd "$SERVER"
exec "$ROOT/.venv/bin/gunicorn" \
    --config "$SERVER/configs/gunicorn_autolabel.py" \
    app.main:app
