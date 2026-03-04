#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── 参数 ────────────────────────────────────────────────────────────────────
MODEL="${LLAMASERVER_MODEL_FILE:-}"
MMPROJ="${LLAMASERVER_MMPROJ_FILE:-}"
HF_REPO="${LLAMASERVER_HF_REPO:-}"

HOST="${LLAMASERVER_BIND_HOST:-127.0.0.1}"
PORT="${LLAMASERVER_PORT:-8080}"
CTX="${LLAMASERVER_CTX:-8192}"
GPU_LAYERS="${LLAMASERVER_GPU_LAYERS:-auto}"
PARALLEL="${LLAMASERVER_PARALLEL:-1}"

TEMP="${LLAMASERVER_TEMP:-1.0}"
TOP_P="${LLAMASERVER_TOP_P:-0.95}"
TOP_K="${LLAMASERVER_TOP_K:-20}"
PRESENCE_PENALTY="${LLAMASERVER_PRESENCE_PENALTY:-1.5}"

# ── 构造命令 ─────────────────────────────────────────────────────────────────
CMD=(llama-server
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$CTX"
    --n-gpu-layers "$GPU_LAYERS"
    --parallel "$PARALLEL"
    --jinja
    --temp "$TEMP"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --presence-penalty "$PRESENCE_PENALTY"
    --log-prefix
)

# 优先用本地文件，否则从 HF 拉
if [[ -n "$MODEL" && -f "$REPO_ROOT/$MODEL" ]]; then
    CMD+=(-m "$REPO_ROOT/$MODEL")
    if [[ -n "$MMPROJ" && -f "$REPO_ROOT/$MMPROJ" ]]; then
        CMD+=(--mmproj "$REPO_ROOT/$MMPROJ")
    fi
    echo "▶ 使用本地模型：$MODEL"
else
    if [[ -z "$HF_REPO" ]]; then
        echo "❌ 错误：未提供模型来源" >&2
        echo "   请设置 LLAMASERVER_HF_REPO=<user>/<repo>，或通过 LLAMASERVER_MODEL_FILE 指定本地模型" >&2
        exit 1
    fi
    CMD+=(-hf "$HF_REPO")
    echo "▶ 从 HuggingFace 加载：$HF_REPO"
fi

echo "▶ 地址：http://${HOST}:${PORT}"
echo "▶ 上下文：${CTX} tokens"
echo ""
echo "${CMD[@]}"
echo ""

exec "${CMD[@]}"
