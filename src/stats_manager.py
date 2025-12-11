from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


STATS_FILE = Path(__file__).resolve().parent.parent / "data" / "stats.json"


def _load_stats() -> Dict[str, Any]:
    if not STATS_FILE.exists():
        return {}
    try:
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_stats(stats: Dict[str, Any]) -> None:
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def update_stats(
    segment: str,
    similarity: float,
    answered: bool,
    in_domain: bool,
) -> None:
    """
    Обновляем агрегированную статистику:
    - total: всего запросов
    - answered: когда бот дал содержательный ответ
    - unknown: когда бот честно сказал "не знаю"
    - false_positive: когда бот ответил, но вопрос был вне домена (по нашему фильтру)
    - similarity_sum: сумма similarity (для оценки среднего confidence)
    """
    stats = _load_stats()
    seg = stats.setdefault(
        segment,
        {
            "total": 0,
            "answered": 0,
            "unknown": 0,
            "false_positive": 0,
            "similarity_sum": 0.0,
            "feedback_good": 0,
            "feedback_bad": 0,
        },
    )

    seg["total"] += 1
    seg["similarity_sum"] += float(similarity or 0.0)

    if answered:
        seg["answered"] += 1
        if not in_domain:
            seg["false_positive"] += 1
    else:
        seg["unknown"] += 1

    _save_stats(stats)


def register_feedback(segment: str, is_good: bool) -> None:
    """
    Регистрируем оценку пользователя:
    - feedback_good / feedback_bad
    """
    stats = _load_stats()
    seg = stats.setdefault(
        segment,
        {
            "total": 0,
            "answered": 0,
            "unknown": 0,
            "false_positive": 0,
            "similarity_sum": 0.0,
            "feedback_good": 0,
            "feedback_bad": 0,
        },
    )

    if is_good:
        seg["feedback_good"] += 1
    else:
        seg["feedback_bad"] += 1

    _save_stats(stats)