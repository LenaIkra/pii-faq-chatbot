from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Пути
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"

SOURCE_XLSX = DATA_PATH / "faqs_raw.xlsx"   # исходник из задания
CHAT_STUDENTS_CSV = DATA_PATH / "faq_chat_students.csv"

# Выходные файлы
KB_APPLICANTS_CSV = DATA_PATH / "faq_kb_applicants.csv"
EMB_APPLICANTS_NPY = DATA_PATH / "faq_embeddings_applicants.npy"

KB_STUDENTS_BASE_CSV = DATA_PATH / "faq_kb_students_base.csv"
KB_STUDENTS_FULL_CSV = DATA_PATH / "faq_kb_students.csv"
EMB_STUDENTS_NPY = DATA_PATH / "faq_embeddings_students.npy"

# Файлы совместимости с текущим FAQBot (одна база)
KB_LEGACY_CSV = DATA_PATH / "faq_kb.csv"
EMB_LEGACY_NPY = DATA_PATH / "faq_embeddings.npy"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def normalize_qa(df: pd.DataFrame) -> pd.DataFrame:
    """Приводим сырую таблицу к виду [question, answer]."""
    df = df.rename(columns=lambda c: str(c).strip())

    # автоопределение колонок с вопросами/ответами
    new_cols: Dict[str, str] = {}
    for c in df.columns:
        name = str(c).strip().lower()
        if "вопрос" in name:
            new_cols[c] = "question"
        elif "ответ" in name:
            new_cols[c] = "answer"
        else:
            new_cols[c] = c
    df = df.rename(columns=new_cols)

    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError(f"Не найдены колонки 'Вопросы/Ответы' в таблице: {df.columns.tolist()}")

    df = df[["question", "answer"]].copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    df = df.dropna(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return df


def load_split_xlsx(path: Path) -> (pd.DataFrame, pd.DataFrame):
    """
    Загружаем Excel и делим на две части:
    - df_applicants: листы для абитуриентов
    - df_students: листы для студентов
    Предполагаем, что в файле есть листы типа 'О программе...' и 'Для студента'.
    Если названия немного отличаются — используем эвристику по подстрокам.
    """
    print(f"[build_kb] Читаю Excel: {path}")
    xls = pd.ExcelFile(path)

    sheets = xls.sheet_names
    print(f"[build_kb] Найдены листы: {sheets}")

    applicants_sheets: List[str] = []
    students_sheets: List[str] = []

    for name in sheets:
        low = name.lower()
        if "студент" in low:
            students_sheets.append(name)
        elif "программ" in low or "абитур" in low:
            applicants_sheets.append(name)
        else:
            # если лист непонятный, по умолчанию кидаем его к абитуриентам
            applicants_sheets.append(name)

    dfs_app = []
    for sheet in applicants_sheets:
        print(f"[build_kb] Лист для абитуриентов: {sheet}")
        df_sheet = pd.read_excel(path, sheet_name=sheet)
        dfs_app.append(normalize_qa(df_sheet))
    df_applicants = pd.concat(dfs_app, ignore_index=True) if dfs_app else pd.DataFrame(columns=["question", "answer"])

    dfs_stud = []
    for sheet in students_sheets:
        print(f"[build_kb] Лист для студентов: {sheet}")
        df_sheet = pd.read_excel(path, sheet_name=sheet)
        dfs_stud.append(normalize_qa(df_sheet))
    df_students = pd.concat(dfs_stud, ignore_index=True) if dfs_stud else pd.DataFrame(columns=["question", "answer"])

    print(f"[build_kb] Всего Q&A для абитуриентов: {len(df_applicants)}")
    print(f"[build_kb] Всего Q&A для студентов (из Excel): {len(df_students)}")

    return df_applicants, df_students


def build_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    """Строим эмбеддинги вопросов."""
    if df.empty:
        raise ValueError("Пустой DataFrame, нечего кодировать в эмбеддинги.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["question"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def maybe_load_chat_students() -> pd.DataFrame:
    """
    Загружаем дополнительную базу для студентов из чата, если есть.
    Ожидается CSV с колонками question,answer.
    """
    if not CHAT_STUDENTS_CSV.exists():
        print(f"[build_kb] Файл с Q&A из чата студентов не найден: {CHAT_STUDENTS_CSV}")
        return pd.DataFrame(columns=["question", "answer"])

    print(f"[build_kb] Загружаю Q&A из чата студентов: {CHAT_STUDENTS_CSV}")
    df = pd.read_csv(CHAT_STUDENTS_CSV)
    df = df.rename(columns=lambda c: str(c).strip())
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Ожидаются колонки 'question' и 'answer' в faq_chat_students.csv")

    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df.dropna(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    print(f"[build_kb] Q&A из чата студентов: {len(df)}")
    return df


def main():
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    if not SOURCE_XLSX.exists():
        raise FileNotFoundError(f"Не найден исходный файл: {SOURCE_XLSX}")

    df_applicants, df_students_base = load_split_xlsx(SOURCE_XLSX)

    # Сохраняем базовую студенческую базу (только из Excel), чтобы было видно "что дали"
    print(f"[build_kb] Сохраняю базу студентов (только Excel) в {KB_STUDENTS_BASE_CSV}")
    df_students_base.to_csv(KB_STUDENTS_BASE_CSV, index=False)

    # Подмешиваем Q&A из чата студентов (если есть)
    df_chat = maybe_load_chat_students()
    if not df_chat.empty:
        df_students_full = (
            pd.concat([df_students_base, df_chat], ignore_index=True)
            .drop_duplicates(subset=["question", "answer"])
            .reset_index(drop=True)
        )
        print(
            f"[build_kb] Студенческая база усилена Q&A из чата: "
            f"{len(df_students_base)} → {len(df_students_full)}"
        )
    else:
        df_students_full = df_students_base.copy()

    # --- Абитуриенты: сохраняем CSV + эмбеддинги ---
    if not df_applicants.empty:
        print(f"[build_kb] Сохраняю базу абитуриентов в {KB_APPLICANTS_CSV}")
        df_applicants.to_csv(KB_APPLICANTS_CSV, index=False)

        print("[build_kb] Строю эмбеддинги для абитуриентов...")
        emb_app = build_embeddings(df_applicants, MODEL_NAME)
        print(f"[build_kb] Сохраняю эмбеддинги абитуриентов в {EMB_APPLICANTS_NPY}")
        np.save(EMB_APPLICANTS_NPY, emb_app)
    else:
        print("[build_kb] ВНИМАНИЕ: база для абитуриентов пустая, эмбеддинги не строю.")

    # --- Студенты: сохраняем усиленную базу + эмбеддинги ---
    if not df_students_full.empty:
        print(f"[build_kb] Сохраняю полную базу студентов в {KB_STUDENTS_FULL_CSV}")
        df_students_full.to_csv(KB_STUDENTS_FULL_CSV, index=False)

        print("[build_kb] Строю эмбеддинги для студентов...")
        emb_stud = build_embeddings(df_students_full, MODEL_NAME)
        print(f"[build_kb] Сохраняю эмбеддинги студентов в {EMB_STUDENTS_NPY}")
        np.save(EMB_STUDENTS_NPY, emb_stud)

        # Файлы совместимости для текущего FAQBot (использует одну базу)
        print(f"[build_kb] Дублирую студенческую базу в legacy-файлы: {KB_LEGACY_CSV}, {EMB_LEGACY_NPY}")
        df_students_full.to_csv(KB_LEGACY_CSV, index=False)
        np.save(EMB_LEGACY_NPY, emb_stud)
    else:
        print("[build_kb] ВНИМАНИЕ: студенческая база пустая, эмбеддинги не строю.")


if __name__ == "__main__":
    main()