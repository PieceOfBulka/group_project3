"""
tools/tool_preprocess.py
Tool 2: Предобработка данных — целевая переменная, признаки, кодирование.
"""

import json
import traceback
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.preprocess")


@tool
def preprocess_data(filepath: str) -> str:
    """
    Предобрабатывает CSV с вакансиями HH.ru: вычисляет целевую переменную (зарплату),
    кодирует категории, создаёт признаки из навыков.
    Аргумент: filepath — путь к CSV файлу.
    """
    logger.info(f"Предобработка | filepath={filepath}")
    try:
        df = pd.read_csv(filepath)

        # Целевая переменная
        if "salary" in df.columns:
            df["salary_target"] = df["salary"].fillna(
                (df.get("salary_from", np.nan) + df.get("salary_to", np.nan)) / 2
            )
        else:
            df["salary_target"] = (df["salary_from"] + df["salary_to"]) / 2
        df = df.dropna(subset=["salary_target"])

        # Опыт
        exp_map = {"noExperience": 0, "between1And3": 2, "between3And6": 4, "moreThan6": 7}
        df["experience_years"] = df["experience"].map(exp_map).fillna(3)

        # Город
        top_cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань"]
        df["is_moscow"] = (df["city"] == "Москва").astype(int)
        df["is_spb"] = (df["city"] == "Санкт-Петербург").astype(int)
        df["is_remote"] = df["city"].str.lower().str.contains("удалённо|удаленно", na=False).astype(int)
        df["is_top_city"] = df["city"].isin(top_cities).astype(int)

        # Занятость и график
        df["employment_score"] = df["employment"].map({"full": 1.0, "part": 0.5, "project": 0.5}).fillna(1.0)
        df["schedule_score"] = df["schedule"].map({"fullDay": 1.0, "remote": 0.7, "flexible": 0.8}).fillna(1.0)

        # Уровень позиции
        level_map = {"junior": 0, "middle": 1, "senior": 2, "lead": 3}
        df["level_score"] = df["level"].map(level_map).fillna(1) if "level" in df.columns else 1
        name_lower = df["name"].str.lower().fillna("")
        level_col = df["level"].fillna("") if "level" in df.columns else pd.Series("", index=df.index)
        df["is_senior"] = (level_col.isin(["senior", "lead"]) | name_lower.str.contains("senior|lead")).astype(int)
        df["is_junior"] = ((level_col == "junior") | name_lower.str.contains("junior|intern")).astype(int)

        # Feature Engineering: категория
        if "category" in df.columns:
            cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
            df = pd.concat([df, cat_dummies], axis=1)

        # Feature Engineering: навыки
        skills_series = df["skills"].fillna("").str.lower().str.replace(" ", "", regex=False)
        df["skills_count"] = skills_series.str.split(r"[;,]").apply(len)

        for skill in ["python", "sql", "docker", "kubernetes", "pytorch", "tensorflow",
                      "javascript", "react", "java", "go", "spark", "airflow",
                      "postgresql", "redis", "kafka", "git", "linux", "pandas"]:
            df[f"skill_{skill}"] = skills_series.str.contains(skill, regex=False).astype(int)

        # Список признаков
        exclude = {"salary_target", "salary_from", "salary_to", "salary", "id", "url",
                   "description", "name", "company", "city", "experience", "employment",
                   "schedule", "skills", "salary_currency", "category", "level"}
        feature_cols = [c for c in df.columns if c not in exclude]

        STATE["df_processed"] = df[feature_cols + ["salary_target"]].copy()
        STATE["feature_cols"] = feature_cols

        result = {
            "status": "ok",
            "rows": len(df),
            "feature_cols": feature_cols,
            "salary_mean": float(df["salary_target"].mean()),
            "salary_min": float(df["salary_target"].min()),
            "salary_max": float(df["salary_target"].max()),
        }
        log_action("preprocess_data", f"Обработано {result['rows']} строк, {len(feature_cols)} признаков")
        logger.info(f"Предобработка завершена | rows={result['rows']} | features={len(feature_cols)} | salary_mean={result['salary_mean']:.0f}")
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error(f"Предобработка ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
