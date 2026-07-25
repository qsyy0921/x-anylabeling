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

MIN_FREE_MB="${AUTOLABEL_MIN_FREE_MB:-8192}"
MAX_UTIL="${AUTOLABEL_MAX_UTIL:-50}"

unset CUDA_VISIBLE_DEVICES
export AUTOLABEL_EFFECTIVE_WORKERS=1
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

echo "Starting Langgao AutoLabel server with one worker and a bounded request queue" >&2
echo "GPU policy: select the least-loaded eligible GPU using utilization and free memory" >&2
echo "Listening on 127.0.0.1:18618 (SSH tunnel required)" >&2

cd "$SERVER"
exec "$ROOT/.venv/bin/gunicorn" \
    --config "$SERVER/configs/gunicorn_autolabel.py" \
    app.main:app
