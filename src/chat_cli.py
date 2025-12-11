from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional

from src.faq_bot import FAQBot
from src.faq_config import FAQ_STUDENTS, FAQ_APPLICANTS
from src.logger import log_interaction


def ask_mode() -> str:
    """
    Жёстко спрашиваем, кто вы: абитуриент или студент.
    Принимаем только 'а'/'a' или 'с'/'c'.
    """
    print("Вы сейчас задаёте вопросы как абитуриент или как студент?")
    print("[а] Абитуриент")
    print("[с] Студент")

    while True:
        mode = input("выбор (а/с): ").strip().lower()

        if mode in ("а", "a"):
            return "applicants"

        if mode in ("с", "c"):
            return "students"

        print('Некорректный ввод. Пожалуйста, введите "а" или "с".\n')


def show_faq_menu(segment: str, bot: FAQBot) -> None:
    """
    Показывает FAQ-меню (категории → вопросы) и сразу отдаёт ответы.
    """
    faq_data: List[Dict[str, Any]] = FAQ_STUDENTS if segment == "students" else FAQ_APPLICANTS

    while True:
        print("\n📘 FAQ — выберите категорию:")
        for i, cat in enumerate(faq_data, start=1):
            print(f"[{i}] {cat['name']}")
        print("[0] Назад\n")

        choice = input("Введите номер категории: ").strip()
        if choice == "0":
            return

        if not choice.isdigit():
            print("Пожалуйста, введите номер.\n")
            continue

        idx = int(choice)
        if not (1 <= idx <= len(faq_data)):
            print("Нет такой категории.\n")
            continue

        category = faq_data[idx - 1]
        questions: List[str] = category["questions"]

        while True:
            print(f"\nКатегория: {category['name']}\n")
            for j, q in enumerate(questions, start=1):
                short_q = q.replace("\n", " ")
                if len(short_q) > 80:
                    short_q = short_q[:77] + "..."
                print(f"[{j}] {short_q}")
            print("[0] Назад\n")

            q_choice = input("Введите номер вопроса: ").strip()
            if q_choice == "0":
                break

            if not q_choice.isdigit():
                print("Пожалуйста, введите номер.\n")
                continue

            q_idx = int(q_choice)
            if not (1 <= q_idx <= len(questions)):
                print("Нет такого вопроса.\n")
                continue

            selected_question = questions[q_idx - 1]
            print(f"\nВы выбрали вопрос:\n> {selected_question}\n")

            answer, match = bot.get_answer(selected_question, debug=False)

            meta = {}
            if match is not None:
                meta = {
                    "matched_question": getattr(match, "question", ""),
                    "similarity": getattr(match, "similarity", None),
                }

            log_interaction("cli", segment, selected_question, answer, meta)
            print("Бот:", answer, "\n")


def main() -> None:
    segment = ask_mode()

    # Чуть снизили порог уверенности, чтобы он чаще давал ответы, если совпадение ок
    bot = FAQBot(segment=segment, sim_threshold=0.5, top_k=3)

    print("\nВиртуальный помощник магистратуры ПИИ.")
    if segment == "students":
        print("Режим: вопросы действующего студента.")
    else:
        print("Режим: вопросы абитуриента.")
    print("Задавайте вопросы в свободной форме.")
    print("Чтобы выйти, напишите: 'выход', 'exit' или 'quit'.")
    print("Чтобы открыть список частых вопросов, напишите: 'faq'.\n")

    while True:
        try:
            user_q = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_q:
            continue

        lower = user_q.lower()

        if lower in ("выход", "exit", "quit", "q"):
            print("Пока! 👋")
            break

        if lower in ("faq", "фак", "меню", "help"):
            show_faq_menu(segment, bot)
            continue

        debug = False
        if lower.startswith("debug:"):
            user_q = user_q[6:].strip()
            debug = True

        answer, match = bot.get_answer(user_q, debug=True)

        meta = {}
        if match is not None:
            meta = {
                "matched_question": getattr(match, "question", ""),
                "similarity": getattr(match, "similarity", None),
            }

        log_interaction("cli", segment, user_q, answer, meta)
        print("\nБот:", answer, "\n")


if __name__ == "__main__":
    main()