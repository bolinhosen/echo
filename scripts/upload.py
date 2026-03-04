"""
upload.py — 将训练好的模型上传到 Hugging Face Hub

用法：
  python upload.py --mode adapter          # 只上传 LoRA adapter（最小，需要 base model）
  python upload.py --mode gguf             # 上传 q4_k_m GGUF（推荐，Ollama/llama.cpp 可用）
  python upload.py --mode gguf_f16         # 上传 f16 GGUF（无损，约 18GB）
  python upload.py --mode merged           # 合并权重后上传完整 16bit 模型
  python upload.py --mode all_gguf         # 上传多种 GGUF 量化格式

环境变量（或直接修改下方常量）：
  HF_TOKEN        Hugging Face write token（https://huggingface.co/settings/tokens）
  HF_USERNAME     你的 HF 用户名
  HF_REPO_NAME    仓库名（默认 echo）
  HF_REPO         完整 repo id，如 myuser/myrepo（设置后忽略上面两个）
"""
from __future__ import annotations

import argparse
import os

import torch
from unsloth import FastLanguageModel

from constants import ADAPTER_DIR, BASE_MODEL as MODEL_NAME, MAX_SEQ_LENGTH

# ──────────────────────────────────────────────────────────────────────────────
# 配置（优先读环境变量，也可以直接改这里）
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")
REPO_NAME   = os.environ.get("HF_REPO_NAME", "echo")
HF_REPO     = os.environ.get("HF_REPO", f"{HF_USERNAME}/{REPO_NAME}")

# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    assert ADAPTER_DIR.exists(), f"LoRA adapter not found: {ADAPTER_DIR}"
    print(f"[*] Loading LoRA adapter: {ADAPTER_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(ADAPTER_DIR),
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit   = False,
        load_in_16bit  = True,
    )
    return model, tokenizer

# ──────────────────────────────────────────────────────────────────────────────
def upload_adapter(model, tokenizer):
    """只推送 LoRA adapter safetensors（最小，~200MB）"""
    print(f"[*] Pushing LoRA adapter → {HF_REPO}")
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"[✓] Done: https://huggingface.co/{HF_REPO}")

def upload_gguf_q4(model, tokenizer):
    """推送 q4_k_m GGUF（约 5GB，推荐 Ollama/llama.cpp 使用）"""
    print(f"[*] Pushing q4_k_m GGUF → {HF_REPO}")
    model.push_to_hub_gguf(
        HF_REPO,
        tokenizer,
        quantization_method = "q4_k_m",
        token = HF_TOKEN,
    )
    print(f"[✓] Done: https://huggingface.co/{HF_REPO}")

def upload_gguf_f16(model, tokenizer):
    """推送 f16 GGUF（约 18GB，无损）"""
    print(f"[*] Pushing f16 GGUF → {HF_REPO}")
    model.push_to_hub_gguf(
        HF_REPO,
        tokenizer,
        quantization_method = "f16",
        token = HF_TOKEN,
    )
    print(f"[✓] Done: https://huggingface.co/{HF_REPO}")

def upload_merged(model, tokenizer):
    """合并权重后推送完整 16bit 模型（约 18GB）"""
    print(f"[*] Pushing merged 16bit model → {HF_REPO}")
    model.push_to_hub_merged(
        HF_REPO,
        tokenizer,
        save_method = "merged_16bit",
        token = HF_TOKEN,
    )
    print(f"[✓] Done: https://huggingface.co/{HF_REPO}")

def upload_all_gguf(model, tokenizer):
    """同时推送多种 GGUF 量化（q4_k_m / q8_0 / q5_k_m）"""
    print(f"[*] Pushing multiple GGUF formats → {HF_REPO}")
    model.push_to_hub_gguf(
        HF_REPO,
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
        token = HF_TOKEN,
    )
    print(f"[✓] Done: https://huggingface.co/{HF_REPO}")

# ──────────────────────────────────────────────────────────────────────────────
MODES = {
    "adapter"  : upload_adapter,
    "gguf"     : upload_gguf_q4,
    "gguf_f16" : upload_gguf_f16,
    "merged"   : upload_merged,
    "all_gguf" : upload_all_gguf,
}

def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to Hugging Face Hub")
    parser.add_argument(
        "--mode", choices=list(MODES.keys()), default="gguf",
        help="上传模式（默认 gguf = q4_k_m）"
    )
    parser.add_argument("--token",    default=None, help="覆盖 HF_TOKEN")
    parser.add_argument("--username", default=None, help="覆盖 HF_USERNAME")
    parser.add_argument("--repo",     default=None, help="覆盖完整 repo id，如 myuser/myrepo")
    args = parser.parse_args()

    global HF_TOKEN, HF_USERNAME, HF_REPO
    if args.token:    HF_TOKEN    = args.token
    if args.username: HF_USERNAME = args.username
    if args.repo:     HF_REPO     = args.repo

    assert HF_TOKEN    != "YOUR_HF_TOKEN",    "请设置 HF_TOKEN 环境变量或 --token 参数"
    assert HF_USERNAME != "YOUR_HF_USERNAME", "请设置 HF_USERNAME 环境变量或 --username 参数"

    model, tokenizer = load_model()
    MODES[args.mode](model, tokenizer)

if __name__ == "__main__":
    main()
