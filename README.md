# pii-faq-chatbot
Чат-бот-помощник координатора для Нетологии
# PII FAQ Chatbot

Виртуальный помощник магистратуры «Прикладной искусственный интеллект» (Нетология + УрФУ).

## Функционал

- Отвечает на частые вопросы:
  - для **абитуриентов** (поступление, стоимость, формат обучения);
  - для **студентов** (расписание, ВКР, практика, справки, дедлайны).
- Поддерживает два режима:
  - **FAQ-меню** (кнопки готовых вопросов);
  - **Свободный вопрос** (поиск по базе с помощью эмбеддингов + TF-IDF).
- Работает:
  - в виде **CLI-чата**;
  - в виде **Telegram-бота**;
  - (опционально) как **web-приложение**.

## Стек

- Python 3.10+
- `sentence-transformers` (эмбеддинги запросов)
- `scikit-learn` (TF-IDF, ранжирование)
- `pandas`, `numpy`
- `python-telegram-bot`
- (опционально) `fastapi`, `uvicorn` для web API

## Установка

```bash
git clone <URL_репозитория>
cd pii-faq-chatbot

python -m venv venv
source venv/bin/activate  # macOS / Linux
# .\venv\Scripts\activate # Windows

pip install -r requirements.txt