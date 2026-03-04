# 训练与推理共用的常量
# 修改此文件后，训练和推理的 system prompt 必须保持一致才能达到最佳效果

from pathlib import Path

# ── 个人信息（修改这里即可） ────────────────────────────────────────────────
NAME      = "roitium"
DEVELOPER = "roitium 科技无限公司"   # identity.json 中 <developer> 的替换值

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    f"你是 {NAME}，一个喜欢技术、ACG 和分享日常的推特用户。"
    "请用简短自然的中文口语风格回复，就像在推特上随手发的消息一样。"
)

# ── 目录 ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent   # 项目根目录
ARCHIVE_ROOT   = ROOT / "twitter_archive"
OUTPUT_DIR     = ROOT / "output"
DATASET_DIR    = OUTPUT_DIR / "dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
ADAPTER_DIR    = CHECKPOINT_DIR / "lora_adapter_final"

# ── 数据流各阶段文件 ──────────────────────────────────────────────────────────
# Step 0: parse_archive.py
TWEETS_PATH             = OUTPUT_DIR / "tweets.json"
REPLIES_MATCHED_PATH    = OUTPUT_DIR / "replies_matched.json"
REPLIES_UNMATCHED_PATH  = OUTPUT_DIR / "replies_unmatched.json"

# Step 1: infer_reply_context.py
REPLIES_INFERRED_PATH   = OUTPUT_DIR / "replies_inferred.json"
REPLIES_INFERRED_JSONL  = OUTPUT_DIR / "replies_inferred.jsonl"
REPLIES_INFERRED_LOG    = OUTPUT_DIR / "replies_inferred_progress.jsonl"

# Step 2: infer_tweet_trigger.py
TWEETS_TRIGGERED_PATH   = OUTPUT_DIR / "tweets_triggered.json"
TWEETS_TRIGGERED_JSONL  = OUTPUT_DIR / "tweets_triggered.jsonl"
TWEETS_TRIGGERED_LOG    = OUTPUT_DIR / "tweets_triggered_progress.jsonl"

# 手写身份问答（受 git 追踪，不放在 output/ 下）
IDENTITY_PATH           = ROOT / "data" / "identity.json"

# Step 4: build_dataset.py
MERGED_DATASET_PATH     = DATASET_DIR / "merged.jsonl"
