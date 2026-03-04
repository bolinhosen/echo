from __future__ import annotations

from unsloth import FastLanguageModel

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from constants import (
    ADAPTER_DIR,
    CHECKPOINT_DIR as OUTPUT_DIR,
    MERGED_DATASET_PATH as DATASET_PATH,
)

# ──────────────────────────────────────────────────────────────────────────────
# 超参
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "unsloth/Qwen3.5-9B"

LORA_R          = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.0

MAX_SEQ_LENGTH  = 2048
BATCH_SIZE      = 1 # 具体看你的 gpu vram，总之我拿 batch=1 跑，peak vram 在 20g 左右
GRAD_ACCUM      = 8
WARMUP_RATIO    = 0.05
NUM_EPOCHS      = 3
LEARNING_RATE   = 4e-4
LR_SCHEDULER    = "cosine"
MAX_STEPS       = -1 
LOGGING_STEPS   = 10
SAVE_STEPS      = 200
SAVE_TOTAL_LIMIT = 3
SEED            = 3407

# ──────────────────────────────────────────────────────────────────────────────
# 1. 加载模型 & Tokenizer
# ──────────────────────────────────────────────────────────────────────────────
print(f"[1/4] Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = False,
    load_in_16bit = True, # Unsloth 强制要求使用 16-bit LoRA，不建议进行 QLora，因为 `higher than normal quantization differences.`
    )
print("[2/4] Injecting LoRA adapters")
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_R,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    use_gradient_checkpointing = "unsloth",
    random_state   = SEED,
    max_seq_length = MAX_SEQ_LENGTH,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. 加载数据集
# ──────────────────────────────────────────────────────────────────────────────
print(f"[3/4] Loading dataset: {DATASET_PATH}")
assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    split="train",
)
print(f"      {len(dataset)} samples loaded")

def format_chat(sample):
    return {"text": tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, add_generation_prompt=False
    )}

processed_dataset = dataset.map(format_chat)

# ──────────────────────────────────────────────────────────────────────────────
# 4. 训练
# ──────────────────────────────────────────────────────────────────────────────
print("[4/4] Starting training")
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = processed_dataset,
    dataset_text_field = "text",
    args = SFTConfig(
        # 数据
        max_seq_length      = MAX_SEQ_LENGTH,
        dataset_num_proc    = 1,

        # 优化
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        learning_rate                = LEARNING_RATE,
        lr_scheduler_type            = LR_SCHEDULER,
        warmup_steps = 10,
        optim                        = "adamw_8bit",

        # 训练长度
        num_train_epochs = NUM_EPOCHS,
        max_steps        = MAX_STEPS, 

        # 日志 & 保存
        logging_steps    = LOGGING_STEPS,
        save_strategy    = "steps",
        save_steps       = SAVE_STEPS,
        save_total_limit = SAVE_TOTAL_LIMIT,
        output_dir       = str(OUTPUT_DIR),

        seed = SEED,
    ),
)

trainer_stats = trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# 5. 保存 LoRA adapter
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter → {ADAPTER_DIR}")
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

print("\n✓ Training complete.")
print(f"  Runtime : {trainer_stats.metrics['train_runtime']:.1f}s")
print(f"  Samples/s: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"  Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
