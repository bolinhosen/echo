"""
用 Gemini 判断独立推文是否有触发语境，
- 有语境 → 生成 inferred_trigger（触发这条推文的上下文/内容）
- 无动机 → 标记 unmotivated: true，inferred_trigger 为 null

输出格式（每条）：
{
  "tweet_id": "...",
  "tweet_text": "...",
  "unmotivated": false,
  "inferred_trigger": "...",
  "model": "...",
  "backend": "..."
}
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import mimetypes
import os
from pathlib import Path
import threading
import time
from typing import Any


PROMPT_TEMPLATE = """你是一个擅长社交媒体语境还原的助手。
任务：判断下面这条独立推文是否有明确的"触发语境"（即作者看到了某内容 / 发生了某件事 / 回应了什么才发出这条推），
并据此二选一输出：

情形 A：推文有合理可推断的触发语境
  → 生成一段简短的"触发内容"，就像它是推友发的帖子、新出的公告、代码报错截图等，能自然引出这条推文的回应。
  → 输出：{{"unmotivated": false, "inferred_trigger": "<10~100 字的触发内容>"}}

情形 B：推文是纯粹的随手发、情绪抒发、日记式喃喃自语，找不到合理触发来源
  → 输出：{{"unmotivated": true, "inferred_trigger": null}}

规则：
1. 不要编造具体人名、链接、事件，除非推文里明确提到。
2. 触发内容语言与推文保持一致（或自然混合）。
3. 宁可判 unmotivated 也不要强行编造牵强的触发内容。
4. 输出必须是 JSON 对象，字段固定为 unmotivated 和 inferred_trigger，不要有其他字段。

推文内容：
{tweet_payload}
"""


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [row for row in data if isinstance(row, dict)]


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model output has no JSON object: {text[:200]}")
    return json.loads(stripped[start : end + 1])


def build_prompt(tweet: dict[str, Any]) -> str:
    payload = {
        "tweet_id": tweet.get("tweet_id"),
        "created_at": tweet.get("created_at"),
        "lang": tweet.get("lang"),
        "text": tweet.get("text", ""),
        "local_media_paths": tweet.get("local_media_paths", []),
    }
    return PROMPT_TEMPLATE.format(
        tweet_payload=json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.as_posix())
    return mime or "application/octet-stream"


def _is_image(path: Path) -> bool:
    return _guess_mime_type(path).startswith("image/")


def build_contents(
    prompt: str,
    archive_root: Path,
    media_paths: list[str],
    types_module: Any,
) -> list[Any]:
    parts = [types_module.Part.from_text(text=prompt)]

    non_image_media: list[str] = []
    for relative_path in media_paths[:4]:
        full_path = archive_root / relative_path
        if not full_path.exists() or not full_path.is_file():
            continue
        if not _is_image(full_path):
            non_image_media.append(relative_path)
            continue
        if full_path.stat().st_size > 15 * 1024 * 1024:
            non_image_media.append(relative_path)
            continue
        parts.append(
            types_module.Part.from_bytes(
                data=full_path.read_bytes(),
                mime_type=_guess_mime_type(full_path),
            )
        )

    if non_image_media:
        parts.append(
            types_module.Part.from_text(
                text=(
                    "以下附件因类型/大小限制未直接输入模型，仅作为上下文参考："
                    + json.dumps(non_image_media, ensure_ascii=False)
                )
            )
        )

    return [types_module.Content(role="user", parts=parts)]


def make_client(args: argparse.Namespace):
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError(
            "google-genai is not installed."
        ) from exc

    if args.backend == "vertex":
        project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        if not project:
            raise ValueError("Vertex backend requires --project or GOOGLE_CLOUD_PROJECT")
        return genai.Client(vertexai=True, project=project, location=args.location)

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("api-key backend requires --api-key or GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def infer_one(
    tweet: dict[str, Any],
    args: argparse.Namespace,
    archive_root: Path,
    types_module: Any,
) -> dict[str, Any]:
    client = make_client(args)
    prompt = build_prompt(tweet)
    media_paths = [p for p in tweet.get("local_media_paths", []) if isinstance(p, str)]
    contents = build_contents(prompt, archive_root, media_paths, types_module)

    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=types_module.GenerateContentConfig(
            temperature=0.35,
            top_p=0.95,
            response_mime_type="application/json",
        ),
    )

    parsed = extract_json_object(response.text or "")

    return {
        "tweet_id": tweet.get("tweet_id"),
        "tweet_text": tweet.get("text"),
        "unmotivated": bool(parsed.get("unmotivated", False)),
        "inferred_trigger": parsed.get("inferred_trigger") or None,
        "model": args.model,
        "backend": args.backend,
        "source_media_paths": media_paths,
    }


def ordered_results(
    ids_in_order: list[str], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [by_id[tid] for tid in ids_in_order if tid in by_id]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer trigger context for standalone tweets using Gemini"
    )
    parser.add_argument("--input", default="output/tweets.json")
    parser.add_argument(
        "--output-json",
        default="output/tweets_triggered.json",
    )
    parser.add_argument(
        "--output-jsonl",
        default="output/tweets_triggered.jsonl",
    )
    parser.add_argument("--archive-root", default="twitter_archive")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument(
        "--backend", choices=["vertex", "api-key"], default="api-key"
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument(
        "--progress-log",
        default="output/tweets_triggered_progress.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_json = Path(args.output_json)
    out_jsonl = Path(args.output_jsonl)
    progress_log = Path(args.progress_log)
    archive_root = Path(args.archive_root)

    tweets = load_json_list(input_path)
    if args.limit > 0:
        tweets = tweets[: args.limit]

    from google.genai import types

    existing = load_existing_results(out_json) if args.resume else []
    by_id: dict[str, dict[str, Any]] = {
        str(item.get("tweet_id") or "").strip(): item
        for item in existing
        if item.get("tweet_id")
    }

    ids_in_order = [str(t.get("tweet_id") or "").strip() for t in tweets]
    pending = [
        t for t in tweets
        if str(t.get("tweet_id") or "").strip()
        and (not args.resume or str(t.get("tweet_id") or "").strip() not in by_id)
    ]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("", encoding="utf-8")

    skipped = len(tweets) - len(pending)
    pending_total = len(pending)
    print(f"resume_loaded={skipped}, pending={pending_total}, total={len(tweets)}")

    with progress_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "start",
                    "total": len(tweets),
                    "pending": pending_total,
                    "resumed": skipped,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # 写入已恢复的结果
    if args.resume:
        out_json.write_text(
            json.dumps(ordered_results(ids_in_order, by_id), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in ordered_results(ids_in_order, by_id):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lock = threading.Lock()
    completed_new = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(infer_one, tweet, args, archive_root, types): tweet
            for tweet in pending
        }

        for future in as_completed(future_map):
            tweet = future_map[future]
            tweet_id = str(tweet.get("tweet_id") or "").strip()
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"[error] tweet_id={tweet_id} error={exc}")
                continue

            with lock:
                by_id[tweet_id] = result

                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                out_json.write_text(
                    json.dumps(
                        ordered_results(ids_in_order, by_id), ensure_ascii=False, indent=2
                    ),
                    encoding="utf-8",
                )

                completed_new += 1
                done_total = skipped + completed_new
                elapsed = time.time() - start_time
                rate = completed_new / elapsed if elapsed > 0 else 0.0
                remaining = max(0, pending_total - completed_new)
                eta_text = format_seconds(remaining / rate if rate > 0 else 0)

                unmotivated_flag = "⬜ skip" if result["unmotivated"] else "✅ has trigger"
                print(
                    f"[{done_total}/{len(tweets)}] {unmotivated_flag} "
                    f"tweet_id={tweet_id} speed={rate:.2f}/s eta={eta_text}"
                )

                with progress_log.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "item_done",
                                "tweet_id": tweet_id,
                                "unmotivated": result["unmotivated"],
                                "done_total": done_total,
                                "total": len(tweets),
                                "remaining_new": remaining,
                                "rate_items_per_sec": round(rate, 4),
                                "eta_hms": eta_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    final_results = ordered_results(ids_in_order, by_id)
    out_json.write_text(
        json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    motivated_count = sum(1 for r in final_results if not r.get("unmotivated"))
    unmotivated_count = sum(1 for r in final_results if r.get("unmotivated"))

    summary = {
        "input_count": len(tweets),
        "output_count": len(final_results),
        "motivated": motivated_count,
        "unmotivated": unmotivated_count,
        "new_completed": completed_new,
        "failed": failed,
        "output_json": str(out_json),
        "output_jsonl": str(out_jsonl),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with progress_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "finish", **summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
