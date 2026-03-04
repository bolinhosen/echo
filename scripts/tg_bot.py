"""
tg_bot.py — Telegram bot，通过 Ollama 与 roitium-echo 对话

依赖：
  pip install python-telegram-bot httpx

环境变量：
  TG_TOKEN        Telegram Bot Token（从 @BotFather 获取）
  OLLAMA_HOST     Ollama 地址，默认 http://localhost:11434
  OLLAMA_MODEL    模型名，默认 roitium-echo

用法：
  TG_TOKEN=xxx python tg_bot.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from telegram import BotCommand, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ──────────────────────────────────────────────────────────────────────────────
TG_TOKEN          = os.environ.get("TG_TOKEN", "YOUR_TG_TOKEN")
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "roitium-echo")

# llama-server (多模态，OpenAI 兼容 API)
BACKEND           = os.environ.get("BACKEND", "ollama")  # "ollama" | "llamaserver"
LLAMASERVER_HOST  = os.environ.get("LLAMASERVER_HOST", "http://localhost:8080")
LLAMASERVER_MODEL = os.environ.get("LLAMASERVER_MODEL", "roitium-echo")

from constants import SYSTEM_PROMPT

# ── 默认生成参数 ──────────────────────────────────────────────────────────────
DEFAULT_OPTIONS: dict[str, float | int] = {
    "temperature"    : 1.0,
    "top_p"          : 0.95,
    "top_k"          : 20,
    "presence_penalty": 1.5,
}

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── 每个 chat_id 独立维护状态
histories:    dict[int, list[dict]]             = defaultdict(list)
options:      dict[int, dict]                   = defaultdict(lambda: dict(DEFAULT_OPTIONS))
msg_id_stack: dict[int, list[tuple[int, int]]]  = defaultdict(list)  # [(user_mid, bot_mid)]

# ── 系统日志（控制台）────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 聊天日志（JSONL，按用户分文件）
# ──────────────────────────────────────────────────────────────────────────────
def _log_path(user_id: int, username: str) -> Path:
    safe_name = re.sub(r"[^\w\-]", "_", username or "unknown")
    return LOG_DIR / f"{user_id}_{safe_name}.jsonl"

def write_log(
    user_id: int, username: str, full_name: str,
    role: str, content: str,
    *, msg_id: int | None = None, extra: dict | None = None,
) -> None:
    record = {
        "ts"       : datetime.now(timezone.utc).isoformat(),
        "user_id"  : user_id,
        "username" : username,
        "full_name": full_name,
        "role"     : role,
        "msg_id"   : msg_id,
        "content"  : content,
    }
    if extra:
        record.update(extra)
    with _log_path(user_id, username).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def strip_think(text: str) -> str:
    """去掉 <think>...</think> 块"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def user_info(update: Update) -> tuple[int, str, str]:
    u = update.effective_user
    return u.id, u.username or "", u.full_name or ""

def make_history(chat_id: int) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + histories[chat_id]

async def chat_with_ollama(chat_id: int, user_msg: str) -> str:
    histories[chat_id].append({"role": "user", "content": user_msg})
    payload = {
        "model"   : OLLAMA_MODEL,
        "messages": make_history(chat_id),
        "stream"  : False,
        "options" : options[chat_id],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw_reply = data["message"]["content"]
    clean     = strip_think(raw_reply)
    histories[chat_id].append({"role": "assistant", "content": raw_reply})
    return clean


async def chat_with_llamaserver(
    chat_id: int,
    user_msg: str,
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
) -> str:
    """通过 llama-server OpenAI 兼容 API 对话，支持图片。"""
    opt = options[chat_id]

    # 构造当前用户消息内容（含图片则使用 content list）
    if image_b64:
        api_user_content: list | str = [
            {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
            {"type": "text", "text": user_msg or "描述这张图片"},
        ]
        hist_user_content = f"[图片] {user_msg}".strip() if user_msg else "[图片]"
    else:
        api_user_content = user_msg
        hist_user_content = user_msg

    # 组装完整消息列表（history 中只存文本，避免 base64 撑爆内存）
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(histories[chat_id])           # 历史（纯文本）
    messages.append({"role": "user", "content": api_user_content})  # 当前轮（含图）

    payload = {
        "model"           : LLAMASERVER_MODEL,
        "messages"        : messages,
        "stream"          : False,
        "temperature"     : opt.get("temperature", 1.0),
        "top_p"           : opt.get("top_p", 0.95),
        "top_k"           : int(opt.get("top_k", 20)),
        "presence_penalty": opt.get("presence_penalty", 1.5),
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{LLAMASERVER_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw_reply = data["choices"][0]["message"]["content"]
    clean = strip_think(raw_reply)

    # 更新文本历史（保留原始含 think）
    histories[chat_id].append({"role": "user",      "content": hist_user_content})
    histories[chat_id].append({"role": "assistant", "content": raw_reply})
    return clean


async def chat(
    chat_id: int,
    user_msg: str,
    image_b64: str | None = None,
    image_mime: str = "image/jpeg",
) -> str:
    """统一入口：根据 BACKEND 调用对应后端。"""
    if BACKEND == "llamaserver":
        return await chat_with_llamaserver(chat_id, user_msg, image_b64, image_mime)
    else:
        if image_b64:
            raise ValueError("Ollama 后端不支持图片，请设置 BACKEND=llamaserver")
        return await chat_with_ollama(chat_id, user_msg)

# ──────────────────────────────────────────────────────────────────────────────
# Command Handlers
# ──────────────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    vision_tip = "📷 发图片（可附文字）→ 让模型分析图片\n" if BACKEND == "llamaserver" else ""
    await update.message.reply_text(
        "👋 我是 roitium-echo！\n\n"
        "直接发消息跟我对话，或用下方命令控制：\n"
        f"{vision_tip}"
        "/undo  — 撤销上一轮对话（含 TG 消息）\n"
        "/clear — 清空对话历史\n"
        "/info  — 查看当前参数\n"
        "/set <参数> <值> — 修改参数\n"
        "/reset — 重置所有参数\n\n"
        "可调参数：temperature · top_p · top_k · presence_penalty"
    )

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    histories[chat_id].clear()
    msg_id_stack[chat_id].clear()
    uid, uname, fname = user_info(update)
    write_log(uid, uname, fname, "system", "/clear", msg_id=update.message.message_id)
    await update.message.reply_text("✅ 对话历史已清空")

async def cmd_undo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    uid, uname, fname = user_info(update)
    stack = msg_id_stack[chat_id]
    hist  = histories[chat_id]

    if not stack:
        await update.message.reply_text("❌ 没有可撤销的对话")
        return

    user_mid, bot_mid = stack.pop()

    # 从历史中移除最后一对 user+assistant
    removed = ""
    if len(hist) >= 2 and hist[-1]["role"] == "assistant" and hist[-2]["role"] == "user":
        removed = hist[-2]["content"]
        hist.pop(); hist.pop()
    elif hist and hist[-1]["role"] == "user":
        removed = hist[-1]["content"]
        hist.pop()

    # 删除 TG 消息
    deleted: list[int] = []
    for mid in (user_mid, bot_mid):
        if mid:
            try:
                await ctx.bot.delete_message(chat_id=chat_id, message_id=mid)
                deleted.append(mid)
            except BadRequest:
                pass

    write_log(uid, uname, fname, "system", f"/undo removed: {removed!r}",
              msg_id=update.message.message_id, extra={"deleted_msg_ids": deleted})

    notice = await update.message.reply_text("↩️ 已撤销上一轮对话")
    await asyncio.sleep(3)
    try:
        await update.message.delete()
        await notice.delete()
    except BadRequest:
        pass

async def cmd_info(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id  = update.effective_chat.id
    opt      = options[chat_id]
    hist_len = len(histories[chat_id])
    model    = LLAMASERVER_MODEL if BACKEND == "llamaserver" else OLLAMA_MODEL
    lines = [f"📊 *当前参数*（模型：`{model}`，后端：`{BACKEND}`）\n"]
    for k, v in opt.items():
        lines.append(f"`{k}` = `{v}`")
    lines.append(f"\n🗂 历史消息数：{hist_len}")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    args = ctx.args
    VALID = {"temperature", "top_p", "top_k", "presence_penalty", "repeat_penalty"}
    if len(args) != 2:
        await update.message.reply_text(
            "用法：`/set <参数> <值>`\n例：`/set temperature 0.7`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    key, val_str = args[0], args[1]
    if key not in VALID:
        await update.message.reply_text(f"❌ 不支持的参数：{key}\n可用：{', '.join(VALID)}")
        return
    try:
        val = int(val_str) if key == "top_k" else float(val_str)
    except ValueError:
        await update.message.reply_text(f"❌ 值必须是数字：{val_str}")
        return
    options[chat_id][key] = val
    uid, uname, fname = user_info(update)
    write_log(uid, uname, fname, "system", f"/set {key}={val}", msg_id=update.message.message_id)
    await update.message.reply_text(f"✅ `{key}` 已设为 `{val}`", parse_mode=ParseMode.MARKDOWN)

async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    options[chat_id] = dict(DEFAULT_OPTIONS)
    uid, uname, fname = user_info(update)
    write_log(uid, uname, fname, "system", "/reset", msg_id=update.message.message_id)
    await update.message.reply_text("✅ 参数已重置为默认值")

async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    global OLLAMA_MODEL
    if not ctx.args:
        await update.message.reply_text(f"当前模型：`{OLLAMA_MODEL}`", parse_mode=ParseMode.MARKDOWN)
        return
    OLLAMA_MODEL = ctx.args[0]
    histories[update.effective_chat.id].clear()
    msg_id_stack[update.effective_chat.id].clear()
    await update.message.reply_text(f"✅ 模型已切换为 `{OLLAMA_MODEL}`，历史已清空", parse_mode=ParseMode.MARKDOWN)

# ──────────────────────────────────────────────────────────────────────────────
# Message Handler
# ──────────────────────────────────────────────────────────────────────────────
async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id   = update.effective_chat.id
    user_text = update.message.text.strip()
    if not user_text:
        return

    uid, uname, fname = user_info(update)
    user_mid = update.message.message_id

    write_log(uid, uname, fname, "user", user_text, msg_id=user_mid)
    logger.info("[%s/%s] → %s", uid, uname or fname, user_text[:80])

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply = await chat(chat_id, user_text)
    except Exception as e:
        logger.exception("chat error")
        await update.message.reply_text(f"❌ 请求失败：{e}")
        return

    if not reply:
        await update.message.reply_text("（模型无输出，请重试）")
        return

    bot_msg = await update.message.reply_text(reply)
    bot_mid = bot_msg.message_id
    msg_id_stack[chat_id].append((user_mid, bot_mid))

    write_log(uid, uname, fname, "assistant", reply, msg_id=bot_mid)
    logger.info("[%s/%s] ← %s", uid, uname or fname, reply[:80])

# ──────────────────────────────────────────────────────────────────────────────
# Image / Sticker / Document Handlers
# ──────────────────────────────────────────────────────────────────────────────
async def _handle_image(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    tg_file,           # GetFile 对象
    mime: str,
    log_label: str,    # 如 "[图片]" / "[贴纸]" / "[文件图片]"
    caption: str = "",
) -> None:
    """通用图片处理：下载 → base64 → 模型 → 回复。"""
    if BACKEND != "llamaserver":
        await update.message.reply_text(
            "❌ 图片功能需要 llama-server 后端\n"
            "启动时设置 BACKEND=llamaserver"
        )
        return

    chat_id  = update.effective_chat.id
    uid, uname, fname = user_info(update)
    user_mid = update.message.message_id

    photo_file  = await tg_file.get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    # 统一转为 PNG，llama-server 的图片解码器不支持 WebP 等格式
    img = Image.open(io.BytesIO(photo_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png"

    log_text = f"{log_label} {caption}".strip() if caption else log_label
    write_log(uid, uname, fname, "user", log_text, msg_id=user_mid)
    logger.info("[%s/%s] → %s", uid, uname or fname, log_text)

    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply = await chat(chat_id, caption, image_b64=image_b64, image_mime=mime)
    except Exception as e:
        logger.exception("chat (image) error")
        await update.message.reply_text(f"❌ 请求失败：{e}")
        return

    if not reply:
        await update.message.reply_text("（模型无输出，请重试）")
        return

    bot_msg = await update.message.reply_text(reply)
    bot_mid = bot_msg.message_id
    msg_id_stack[chat_id].append((user_mid, bot_mid))

    write_log(uid, uname, fname, "assistant", reply, msg_id=bot_mid)
    logger.info("[%s/%s] ← %s", uid, uname or fname, reply[:80])


async def on_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """压缩图片（JPEG）。"""
    caption = (update.message.caption or "").strip()
    photo   = update.message.photo[-1]           # 最高分辨率
    await _handle_image(update, ctx, photo, "image/jpeg", "[图片]", caption)


async def on_sticker(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """贴纸 / 表情包（静态 WebP）。"""
    sticker = update.message.sticker
    if sticker.is_animated or sticker.is_video:
        await update.message.reply_text("动态贴纸/表情暂时不支持分析 😅")
        return
    await _handle_image(update, ctx, sticker, "image/webp", "[贴纸]")


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """以文件形式发送的图片（不压缩原图）。"""
    doc = update.message.document
    mime = doc.mime_type or "image/jpeg"
    caption = (update.message.caption or "").strip()
    await _handle_image(update, ctx, doc, mime, "[文件图片]", caption)


async def on_animation(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """GIF 动图（Telegram 内部存为 MP4，视觉模型暂不支持）。"""
    await update.message.reply_text("GIF 动图暂时不支持分析，发静止图片或贴纸吧 😅")


# ──────────────────────────────────────────────────────────────────────────────
# 注册 Command Panel
# ──────────────────────────────────────────────────────────────────────────────
async def post_init(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start", "开始 / 查看帮助"),
        BotCommand("undo",  "撤销上一轮对话并删除消息"),
        BotCommand("clear", "清空对话历史"),
        BotCommand("info",  "查看当前参数"),
        BotCommand("set",   "修改参数，如 /set temperature 0.7"),
        BotCommand("reset", "重置所有参数为默认值"),
        BotCommand("model", "切换模型，如 /model roitium-echo"),
    ])

# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    assert TG_TOKEN != "YOUR_TG_TOKEN", "请设置 TG_TOKEN 环境变量"

    app = (
        Application.builder()
        .token(TG_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("undo",  cmd_undo))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("info",  cmd_info))
    app.add_handler(CommandHandler("set",   cmd_set))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(MessageHandler(filters.PHOTO,                   on_photo))
    app.add_handler(MessageHandler(filters.Sticker.ALL,             on_sticker))
    app.add_handler(MessageHandler(filters.Document.IMAGE,          on_document))
    app.add_handler(MessageHandler(filters.ANIMATION,               on_animation))

    backend_info = f"llamaserver={LLAMASERVER_HOST}" if BACKEND == "llamaserver" else f"ollama={OLLAMA_HOST}"
    logger.info("Bot started | backend=%s | %s", BACKEND, backend_info)
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
