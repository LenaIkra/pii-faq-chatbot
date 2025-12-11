from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from pathlib import Path

import pandas as pd

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from src.faq_bot import FAQBot
from src.faq_config import FAQ_STUDENTS, FAQ_APPLICANTS
from src.logger import log_interaction
from src.stats_manager import register_feedback


# (Опционально) заранее инициализированные боты — пока не используем,
# но можно будет подключить позже, чтобы не пересоздавать модели.
BOTS = {
    "students": FAQBot(segment="students", sim_threshold=0.5, top_k=3),
    "applicants": FAQBot(segment="applicants", sim_threshold=0.5, top_k=3),
}


def main_keyboard() -> ReplyKeyboardMarkup:
    """Главная клавиатура под строкой ввода."""
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("Я студент"), KeyboardButton("Я абитуриент")],
            [KeyboardButton("FAQ"), KeyboardButton("Свободный вопрос")],
        ],
        resize_keyboard=True,
    )


def feedback_keyboard(segment: str) -> InlineKeyboardMarkup:
    """Клавиатура для оценки ответа (👍 / 👎)."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text="👍 Ответ помог", callback_data=f"fb:good:{segment}"
                ),
                InlineKeyboardButton(
                    text="👎 Не помог", callback_data=f"fb:bad:{segment}"
                ),
            ]
        ]
    )


def build_faq_keyboard_from_csv(segment: str, limit: int = 10) -> InlineKeyboardMarkup:
    """
    Строим FAQ-меню из CSV:
      - для студентов:     data/faq_kb_students.csv
      - для абитуриентов:  data/faq_kb_applicants.csv
    Показываем первые `limit` вопросов.
    """
    base_dir = Path(__file__).resolve().parent.parent / "data"
    if segment == "applicants":
        kb_csv = base_dir / "faq_kb_applicants.csv"
    else:
        kb_csv = base_dir / "faq_kb_students.csv"

    df = pd.read_csv(kb_csv)

    keyboard: list[list[InlineKeyboardButton]] = []

    for idx, row in df.head(limit).iterrows():
        question = str(row["question"])
        # немного укоротим текст кнопки
        if len(question) > 60:
            btn_text = question[:57] + "..."
        else:
            btn_text = question

        keyboard.append(
            [
                InlineKeyboardButton(
                    text=btn_text,
                    callback_data=f"faq_q:{segment}:{idx}",
                )
            ]
        )

    return InlineKeyboardMarkup(keyboard)


def detect_segment_from_text(text: str) -> str:
    """Простейший детектор: студент/абитуриент по тексту кнопки."""
    t = text.lower()
    if "абит" in t:
        return "applicants"
    if "студ" in t:
        return "students"
    return "students"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка /start."""
    await update.message.reply_text(
        "Привет! Я виртуальный помощник магистратуры «Прикладной искусственный интеллект».\n\n"
        "Сначала выберите, кто вы:",
        reply_markup=main_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка /help."""
    await update.message.reply_text(
        "Я могу:\n"
        "• отвечать на частые вопросы по кнопке FAQ;\n"
        "• отвечать на произвольные вопросы (Свободный вопрос);\n\n"
        "Сначала выберите, кто вы: студент или абитуриент.",
        reply_markup=main_keyboard(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка обычных текстовых сообщений.

    Логика:
      1) "я студент" / "я абитуриент" → смена режима, НЕ спрашиваем RAG
      2) "FAQ" / "частые вопросы"      → открываем меню FAQ из CSV
      3) "Свободный вопрос"           → просим ввести текст вопроса
      4) всё остальное                → вопрос в RAG (FAQBot)
    """
    if not update.message:
        return

    user = update.effective_user
    text = (update.message.text or "").strip()

    if not text:
        await update.message.reply_text("Пожалуйста, введите текст вопроса.")
        return

    lower = text.lower()

    # 1️⃣ Служебные фразы — выбор сегмента
    if "абитуриент" in lower:
        context.user_data["segment"] = "applicants"
        await update.message.reply_text(
            "Окей, переключаюсь в режим абитуриента. "
            "Теперь можно задавать вопросы про поступление, программу и приём.",
            reply_markup=main_keyboard(),
        )
        return

    if "студент" in lower:
        context.user_data["segment"] = "students"
        await update.message.reply_text(
            "Окей, переключаюсь в режим студента. "
            "Теперь можно задавать вопросы про учёбу, сессии, практику и ВКР.",
            reply_markup=main_keyboard(),
        )
        return

    # 2️⃣ Раздел FAQ (кнопка 'FAQ' / 'Частые вопросы')
    if "faq" in lower or "частые вопрос" in lower:
        segment = context.user_data.get("segment", "students")
        if segment not in ("students", "applicants"):
            segment = "students"

        kb = build_faq_keyboard_from_csv(segment)
        await update.message.reply_text(
            "Вот список часто задаваемых вопросов. Выбери интересующий:",
            reply_markup=kb,
        )
        return

    # 2.1️⃣ Кнопка "Свободный вопрос"
    if "свободный вопрос" in lower:
        await update.message.reply_text(
            "Окей, это режим свободного вопроса.\n"
            "Напишите, пожалуйста, свой вопрос текстом — я постараюсь найти ответ в базе."
        )
        return

    # 3️⃣ Всё остальное — обычный вопрос → в RAG FAQBot
    segment = context.user_data.get("segment", "students")
    if segment not in ("students", "applicants"):
        segment = "students"

    bot_key = f"faq_{segment}"
    faq_bot: FAQBot | None = context.bot_data.get(bot_key)
    if faq_bot is None:
        faq_bot = FAQBot(segment=segment)
        context.bot_data[bot_key] = faq_bot

    answer, match = faq_bot.get_answer(text, debug=False)

    # Логируем запрос/ответ
    meta = {
        "user_id": user.id if user else None,
    }
    if match is not None:
        meta["matched_question"] = match.question
        meta["similarity"] = float(match.similarity)
    else:
        meta["matched_question"] = None
        meta["similarity"] = None

    # Важно: log_interaction — позиционные аргументы
    log_interaction("telegram", segment, text, answer, meta)

    # Ответ пользователю
    await update.message.reply_text(answer)

    # Кнопки для оценки ответа
    await update.message.reply_text(
        "Ответ был полезен?",
        reply_markup=feedback_keyboard(segment),
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка нажатий на inline-кнопки:
      - faq_q:segment:index  — выбор конкретного FAQ-вопроса из CSV
      - fb:good:segment / fb:bad:segment — фидбек на ответ
    """
    query = update.callback_query
    if not query:
        return

    data = (query.data or "").strip()

    # 1️⃣ Вопрос из FAQ-меню: faq_q:segment:index
    if data.startswith("faq_q:"):
        try:
            _, segment, idx_str = data.split(":")
            idx = int(idx_str)
        except ValueError:
            await query.answer("Ошибка формата данных.")
            return

        base_dir = Path(__file__).resolve().parent.parent / "data"
        if segment == "applicants":
            kb_csv = base_dir / "faq_kb_applicants.csv"
        else:
            kb_csv = base_dir / "faq_kb_students.csv"

        try:
            df = pd.read_csv(kb_csv)
        except Exception:
            await query.answer("Не удалось прочитать базу FAQ.")
            return

        if idx < 0 or idx >= len(df):
            await query.answer("Этот вопрос больше недоступен.")
            return

        row = df.iloc[idx]
        question = str(row["question"])
        answer = str(row["answer"])

        # Меняем текст сообщения на вопрос+ответ
        try:
            await query.message.edit_text(f"❓ {question}\n\n💬 {answer}")
        except Exception:
            # если не получилось отредактировать — просто новое сообщение
            await query.message.reply_text(f"❓ {question}\n\n💬 {answer}")

        await query.answer()
        return

    # 2️⃣ Фидбек: fb:good:segment / fb:bad:segment
    if data.startswith("fb:"):
        try:
            _, fb_type, segment = data.split(":")
        except ValueError:
            await query.answer("Некорректный формат ответа.")
            return

        is_good = fb_type == "good"
        register_feedback(segment, is_good)

        # убираем кнопки фидбека
        try:
            await query.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

        await query.answer("Спасибо за оценку!")
        return

    # 3️⃣ Всё остальное
    await query.answer("Неизвестное действие")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Простой логгер ошибок в консоль."""
    print(f"[TelegramBot] Ошибка: {context.error}")


def run() -> None:
    """Запуск Telegram-бота."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Нужно установить переменную окружения TELEGRAM_BOT_TOKEN с токеном бота."
        )

    application = ApplicationBuilder().token(token).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_error_handler(error_handler)

    print("[TelegramBot] Бот запущен. Ожидаю сообщения...")
    application.run_polling()


if __name__ == "__main__":
    run()