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
任务：根据“回复推文”内容，推测它最可能在回复的“原推内容”。

请严格遵守：
1) 只根据输入信息做高置信推测，不要编造具体事实（人名、链接、具体事件细节）除非回复明确提到。
2) 推测的原推要自然、像真实用户发帖，不要解释过程。
3) 优先保持与回复相同语言（若回复是中英混合，可中英混合）。
4) 推测原推长度控制在 10~120 字。
5) 输出必须是 JSON 对象，字段固定为：
    - inferred_original_tweet: string

输入回复信息：
{reply_payload}
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
        raise ValueError(f"Model output does not contain JSON object: {text[:200]}")

    return json.loads(stripped[start : end + 1])


def build_prompt(reply: dict[str, Any]) -> str:
    import re
    text = reply.get("text") or ""
    text_no_mentions = re.sub(r'@\w+\s*', '', text).strip()
    
    payload = {
        "tweet_id": reply.get("tweet_id"),
        "created_at": reply.get("created_at"),
        "lang": reply.get("lang"),
        "text": text_no_mentions,
        "in_reply_to_screen_name": reply.get("in_reply_to_screen_name"),
        "local_media_paths": reply.get("local_media_paths", []),
    }
    return PROMPT_TEMPLATE.format(reply_payload=json.dumps(payload, ensure_ascii=False, indent=2))


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.as_posix())
    return mime or "application/octet-stream"


def _is_image(path: Path) -> bool:
    mime = _guess_mime_type(path)
    return mime.startswith("image/")


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

        file_size = full_path.stat().st_size
        if file_size > 15 * 1024 * 1024:
            non_image_media.append(relative_path)
            continue

        mime_type = _guess_mime_type(full_path)
        data = full_path.read_bytes()
        parts.append(types_module.Part.from_bytes(data=data, mime_type=mime_type))

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
            "google-genai is not installed. Run: python scripts/install_vertex_ai_deps.py"
        ) from exc

    if args.backend == "vertex":
        project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        if not project:
            raise ValueError("Vertex backend requires --project or GOOGLE_CLOUD_PROJECT")
        location = args.location
        client = genai.Client(vertexai=True, project=project, location=location)
        return client

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("api-key backend requires --api-key or GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    return client


def infer_one(
    reply: dict[str, Any],
    args: argparse.Namespace,
    archive_root: Path,
    types_module: Any,
) -> dict[str, Any]:
    client = make_client(args)
    prompt = build_prompt(reply)
    media_paths = reply.get("local_media_paths", [])
    media_paths = [p for p in media_paths if isinstance(p, str)]
    contents = build_contents(
        prompt,
        archive_root=archive_root,
        media_paths=media_paths,
        types_module=types_module,
    )

    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=types_module.GenerateContentConfig(
            temperature=0.35,
            top_p=0.95,
            response_mime_type="application/json",
        ),
    )

    text = response.text or ""
    parsed = extract_json_object(text)

    return {
        "tweet_id": reply.get("tweet_id"),
        "reply_text": reply.get("text"),
        "in_reply_to_status_id": reply.get("in_reply_to_status_id"),
        "inferred_original_tweet": parsed.get("inferred_original_tweet", ""),
        "model": args.model,
        "backend": args.backend,
        "source_media_paths": media_paths,
    }


def ordered_results(reply_ids_in_order: list[str], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [by_id[tweet_id] for tweet_id in reply_ids_in_order if tweet_id in by_id]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer most likely parent tweet for context-missing replies using Gemini"
    )
    parser.add_argument(
        "--input",
        default="output/replies_unmatched.json",
        help="Input replies json path",
    )
    parser.add_argument(
        "--output-json",
        default="output/replies_inferred.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-jsonl",
        default="output/replies_inferred.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--archive-root",
        default="twitter_archive",
        help="Archive root folder, used to resolve local media paths",
    )
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    parser.add_argument(
        "--backend",
        choices=["vertex", "api-key"],
        default="api-key",
        help="Auth backend: vertex (GCP project) or api-key",
    )
    parser.add_argument("--project", default=None, help="GCP project for vertex backend")
    parser.add_argument("--location", default="us-central1", help="GCP location for vertex backend")
    parser.add_argument("--api-key", default=None, help="Gemini API key for api-key backend")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0 = all)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent request workers")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output-json (default: enabled)",
    )
    parser.add_argument(
        "--progress-log",
        default="output/replies_inferred_progress.jsonl",
        help="Progress log output path (JSONL, one line per completed item)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_json = Path(args.output_json)
    out_jsonl = Path(args.output_jsonl)
    progress_log = Path(args.progress_log)
    archive_root = Path(args.archive_root)

    replies = load_json_list(input_path)
    if args.limit > 0:
        replies = replies[: args.limit]

    from google.genai import types

    existing = load_existing_results(out_json) if args.resume else []
    by_id: dict[str, dict[str, Any]] = {}
    for item in existing:
        tweet_id = str(item.get("tweet_id") or "").strip()
        if tweet_id:
            by_id[tweet_id] = item

    reply_ids_in_order = [str(reply.get("tweet_id") or "").strip() for reply in replies]
    pending_replies = []
    for reply in replies:
        tweet_id = str(reply.get("tweet_id") or "").strip()
        if not tweet_id:
            continue
        if args.resume and tweet_id in by_id:
            continue
        pending_replies.append(reply)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("", encoding="utf-8")

    start_time = time.time()
    pending_total = len(pending_replies)

    with progress_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "start",
                    "total": len(replies),
                    "pending": pending_total,
                    "resumed": len(replies) - pending_total,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if args.resume:
        out_json.write_text(
            json.dumps(ordered_results(reply_ids_in_order, by_id), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in ordered_results(reply_ids_in_order, by_id):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(replies)
    skipped = len(replies) - len(pending_replies)
    print(f"resume_loaded={skipped}, pending={len(pending_replies)}, total={total}")

    lock = threading.Lock()
    completed_new = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(infer_one, reply, args, archive_root, types): reply
            for reply in pending_replies
        }

        for future in as_completed(future_map):
            reply = future_map[future]
            tweet_id = str(reply.get("tweet_id") or "").strip()
            try:
                inferred = future.result()
            except Exception as exc:
                failed += 1
                print(f"[error] tweet_id={tweet_id} error={exc}")
                continue

            with lock:
                by_id[tweet_id] = inferred

                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(inferred, ensure_ascii=False) + "\n")

                out_json.write_text(
                    json.dumps(ordered_results(reply_ids_in_order, by_id), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                completed_new += 1
                done_total = skipped + completed_new
                elapsed = time.time() - start_time
                rate = completed_new / elapsed if elapsed > 0 else 0.0
                remaining_new = max(0, pending_total - completed_new)
                eta_seconds = remaining_new / rate if rate > 0 else 0
                eta_text = format_seconds(eta_seconds)

                with progress_log.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "item_done",
                                "tweet_id": tweet_id,
                                "done_total": done_total,
                                "total": total,
                                "new_completed": completed_new,
                                "pending_total": pending_total,
                                "remaining_new": remaining_new,
                                "rate_items_per_sec": round(rate, 4),
                                "eta_seconds": round(eta_seconds, 2),
                                "eta_hms": eta_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                print(
                    f"[{done_total}/{total}] inferred tweet_id={tweet_id} "
                    f"speed={rate:.2f}/s eta={eta_text}"
                )

    final_results = ordered_results(reply_ids_in_order, by_id)
    out_json.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_count": len(replies),
                "output_count": len(final_results),
                "resumed_count": skipped,
                "new_completed": completed_new,
                "failed": failed,
                "output_json": str(out_json),
                "output_jsonl": str(out_jsonl),
                "progress_log": str(progress_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    with progress_log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "event": "finish",
                    "input_count": len(replies),
                    "output_count": len(final_results),
                    "resumed_count": skipped,
                    "new_completed": completed_new,
                    "failed": failed,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
