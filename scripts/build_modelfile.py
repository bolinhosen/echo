"""
根据 constants.py 中的 SYSTEM_PROMPT 生成项目根目录的 Modelfile。
每次修改 SYSTEM_PROMPT 后运行一次，再重新 ollama create。

用法：
  python scripts/build_modelfile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constants import ROOT, SYSTEM_PROMPT

GGUF_PATH = "./models/Qwen3.5-9B.Q4_K_M.gguf"

content = (
    f"FROM {GGUF_PATH}\n"
    "\n"
    'TEMPLATE """<|im_start|>system\n'
    "{{ .System }}<|im_end|>\n"
    "{{ range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
    '"""\n'
    "\n"
    f'SYSTEM "{SYSTEM_PROMPT}"\n'
    "\n"
    "PARAMETER temperature 1\n"
    "PARAMETER top_p 0.95\n"
    "PARAMETER top_k 20\n"
    "PARAMETER presence_penalty 1.5\n"
    'PARAMETER stop "<|im_end|>"\n'
    'PARAMETER stop "<|im_start|>"\n'
    'PARAMETER stop "<|endoftext|>"\n'
)

out = ROOT / "Modelfile"
out.write_text(content, encoding="utf-8")
print(f"Written: {out}")
print(f"SYSTEM: {SYSTEM_PROMPT}")
