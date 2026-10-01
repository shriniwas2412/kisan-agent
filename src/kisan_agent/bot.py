"""Telegram front-end:  TELEGRAM_BOT_TOKEN=... kisan-bot

Text → agent. Photo (+ optional caption) → agent with the image attached, so it can call
diagnose_leaf. Voice notes get a polite "please type" for now.
"""

from __future__ import annotations

import logging
import os

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .agent import KisanAgent
from .ratelimit import DailyQuota

log = logging.getLogger("kisan-bot")

WELCOME = (
    "🙏 Namaste! Main Kisan Mitra hoon.\n\n"
    "• Patti ki photo bhejiye: bimari pehchanne ki koshish karunga\n"
    "• Mandi bhav poochiye: \"Sehore mein soyabean ka bhav?\"\n"
    "• Mausam / chhidkav: \"Kal Vidisha mein spray kar sakte hain?\"\n\n"
    "Dawai aur matra hamesha apne KVK se confirm karein."
)
LIMIT_REACHED = "Aaj ke sawal poore ho gaye 🙏 Kal phir poochiye."
VOICE_NOT_SUPPORTED = "Abhi main voice note nahi sun sakta. Kripya likh kar poochiye 🙏"
ERROR = "Maaf kijiye, abhi kuch gadbad ho gayi. Thodi der baad koshish karein."

agent = KisanAgent()
quota = DailyQuota(limit=int(os.getenv("DAILY_QUESTION_LIMIT", "20")))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not quota.allow(msg.from_user.id):
        await msg.reply_text(LIMIT_REACHED)
        return
    await context.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)

    images = None
    text = msg.text or msg.caption or ""
    if msg.photo:
        photo = await msg.photo[-1].get_file()  # largest size Telegram offers
        images = {"photo1": bytes(await photo.download_as_bytearray())}
        text = text or "Is patti mein kya bimari hai?"

    try:
        trace = await agent.ask(text, images)
    except Exception:
        log.exception("agent failed")
        await msg.reply_text(ERROR)
        return
    # Log tools/tokens/latency only; never the farmer's message or photo.
    log.info("tools=%s tokens=%d/%d latency=%.1fs", [c["name"] for c in trace.tool_calls],
             trace.input_tokens, trace.output_tokens, trace.latency_s)
    await msg.reply_text(trace.answer or ERROR)


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(VOICE_NOT_SUPPORTED)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.PHOTO, answer))
    app.run_polling()


if __name__ == "__main__":
    main()
