#!/bin/bash
set -e

MODEL_ID="${VIBEVOICE_MODEL:-microsoft/VibeVoice-ASR}"
PORT="${VIBEVOICE_PORT:-8001}"
GPU_UTIL="${VIBEVOICE_GPU_UTIL:-0.30}"

echo "[vibevoice] locating model: $MODEL_ID"
MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
import sys
try:
    p = snapshot_download('$MODEL_ID', local_files_only=True)
    print(p)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
echo "[vibevoice] model path: $MODEL_PATH"

# Generate tokenizer files once (idempotent)
if [ ! -f "$MODEL_PATH/tokenizer.json" ]; then
    echo "[vibevoice] generating tokenizer files..."
    python3 -m vllm_plugin.tools.generate_tokenizer_files --output "$MODEL_PATH"
fi

echo "[vibevoice] starting vLLM on port $PORT (gpu_util=$GPU_UTIL)"
exec vllm serve "$MODEL_PATH" \
    --served-model-name vibevoice \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 16 \
    --max-model-len 65536 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --chat-template-content-format openai \
    --allowed-local-media-path /tmp \
    --port "$PORT"
