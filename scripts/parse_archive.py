"""
从 Twitter 存档解析推文，输出供后续流水线使用的 JSON 文件。

产出：
  output/tweets.json            — 独立推文（无父推）
  output/replies_matched.json   — 父推在存档内的回复，含 parent_text 字段
  output/replies_unmatched.json — 父推不在存档内的孤立回复

用法：
  python scripts/parse_archive.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from constants import (
    ARCHIVE_ROOT as DEFAULT_ARCHIVE_ROOT,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    REPLIES_MATCHED_PATH,
    REPLIES_UNMATCHED_PATH,
    TWEETS_PATH,
)


# ── 存档解析 ──────────────────────────────────────────────────────────────────

def _extract_json_from_ytd_js(file_path: Path) -> Any:
    """Twitter 存档 JS 文件格式：window.YTD.xxx.partN = [...]，提取其中的数组。"""
    raw = file_path.read_text(encoding="utf-8")
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Cannot parse YTD JS array from {file_path}")
    return json.loads(raw[start : end + 1])


def _load_tweets(archive_root: Path) -> list[dict[str, Any]]:
    tweets_file = archive_root / "data" / "tweets.js"
    if not tweets_file.exists():
        raise FileNotFoundError(f"tweets.js not found: {tweets_file}")
    rows = _extract_json_from_ytd_js(tweets_file)
    return [item["tweet"] for item in rows if isinstance(item, dict) and isinstance(item.get("tweet"), dict)]


# ── 过滤与分类 ────────────────────────────────────────────────────────────────

def _is_repost(tweet: dict[str, Any]) -> bool:
    if tweet.get("retweeted") is True:
        return True
    return str(tweet.get("full_text") or "").startswith("RT @")


def _is_reply(tweet: dict[str, Any]) -> bool:
    return bool(tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id"))


# ── 记录构建 ──────────────────────────────────────────────────────────────────

def _to_record(tweet: dict[str, Any]) -> dict[str, Any]:
    tweet_id = str(tweet.get("id_str") or tweet.get("id") or "")
    return {
        "tweet_id":                tweet_id,
        "created_at":              tweet.get("created_at"),
        "text":                    tweet.get("full_text", ""),
        "lang":                    tweet.get("lang"),
        "in_reply_to_status_id":   tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id"),
        "in_reply_to_screen_name": tweet.get("in_reply_to_screen_name"),
        "favorite_count":          int(tweet.get("favorite_count") or 0),
        "retweet_count":           int(tweet.get("retweet_count") or 0),
    }


# ── IO ────────────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run(archive_root: Path, output_dir: Path) -> dict[str, int]:
    tweets = _load_tweets(archive_root)

    # 第一遍：建立全量 id → text 索引（用于父推检索）
    id_to_text: dict[str, str] = {}
    for tweet in tweets:
        tid = str(tweet.get("id_str") or tweet.get("id") or "")
        if tid:
            id_to_text[tid] = tweet.get("full_text", "")

    originals:        list[dict[str, Any]] = []
    with_context:     list[dict[str, Any]] = []
    missing_context:  list[dict[str, Any]] = []

    for tweet in tweets:
        if _is_repost(tweet):
            continue

        record = _to_record(tweet)

        if not _is_reply(tweet):
            originals.append(record)
            continue

        parent_id = record["in_reply_to_status_id"] or ""
        parent_text = id_to_text.get(str(parent_id), "")

        if parent_text:
            with_context.append({**record, "parent_tweet_id": parent_id, "parent_text": parent_text})
        else:
            missing_context.append(record)

    for lst in (originals, with_context, missing_context):
        lst.sort(key=lambda r: str(r.get("created_at") or ""))

    _write_json(output_dir / TWEETS_PATH.name,            originals)
    _write_json(output_dir / REPLIES_MATCHED_PATH.name,    with_context)
    _write_json(output_dir / REPLIES_UNMATCHED_PATH.name,  missing_context)

    stats = {
        "total_tweets":           len(tweets),
        "original_tweets":        len(originals),
        "replies_with_context":   len(with_context),
        "replies_missing_context": len(missing_context),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tweets from Twitter archive")
    parser.add_argument("archive_root", nargs="?", default=str(DEFAULT_ARCHIVE_ROOT),
                        help=f"Path to Twitter archive root (default: {DEFAULT_ARCHIVE_ROOT})")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    stats = run(
        archive_root=Path(args.archive_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
