"""
benchmark_llms.py
Сравнение LLM-моделей по качеству генерации кода для ML-пайплайна.
Запуск: python benchmark_llms.py
Результат: llm_comparison.csv
"""

import os
import time
import csv
import json
import traceback
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

MODELS = [
    {"name": "GPT-OSS 120B", "id": "openai/gpt-oss-120b:free", "tier": "free"},
    {"name": "GLM-4.5 Air", "id": "z-ai/glm-4.5-air:free", "tier": "free"},
    {"name": "DeepSeek V4 Flash", "id": "deepseek/deepseek-v4-flash", "tier": "paid"},
    {"name": "MiniMax M2.7", "id": "minimax/minimax-m2.7", "tier": "paid"},
]

DATA_FILE = "data/hh_it_5000_final.csv"

PROMPTS = {
    "eda": (
        "eda",
        f"""Ты — Data Engineer. Напиши Python-код для загрузки и анализа CSV файла с вакансиями HH.ru.

ЗАДАЧА:
1. Загрузи CSV файл по пути: {DATA_FILE}
2. Сохрани результат в переменную `result` как словарь со следующими ключами:
   - "rows": количество строк (int)
   - "columns": список названий колонок (list)
   - "salary_columns": список колонок связанных с зарплатой (list)
   - "missing_values": словарь {{колонка: кол-во пропусков}} только для колонок с пропусками
   - "sample": первые 3 строки — df.head(3).to_dict('records')
   - "summary": краткое описание датасета (строка)

ВАЖНО:
- import pandas as pd
- Возвращай только Python-код, без пояснений, без markdown-блоков
""",
    ),
    "preprocess": (
        "preprocess",
        f"""Ты — опытный ML-инженер. Напиши Python-код для предобработки датасета вакансий с HH.ru.

ФАЙЛ: {DATA_FILE}

ЗАДАЧА:
1. Загрузи CSV, вычисли salary_target = среднее(salary_from, salary_to)
2. Удали строки где salary_target NaN
3. Создай experience_years: noExperience→0, between1And3→2, between3And6→4, moreThan6→7
4. Создай is_moscow, is_spb (бинарные по колонке city)
5. Создай skills_count — количество навыков (разделитель ";")
6. Создай skill_python = 1 если "Python" в skills
7. Сохрани:
   - `df_processed` — итоговый DataFrame
   - `feature_cols` — список: ["experience_years","is_moscow","is_spb","skills_count","skill_python"]
   - `result` — {{"status":"ok","rows":int,"feature_cols":list,"salary_mean":float}}

import pandas as pd, import numpy as np
Возвращай только Python-код без пояснений и без markdown-блоков.
""",
    ),
    "train": (
        "train",
        """Ты — ML-инженер. Напиши Python-код для обучения и сравнения трёх моделей регрессии.

Переменные уже доступны:
- `df` — pandas DataFrame с колонками salary_target и ["experience_years","is_moscow","is_spb","skills_count","skill_python"]
- `feature_cols` = ["experience_years","is_moscow","is_spb","skills_count","skill_python"]

ЗАДАЧА:
1. X = df[feature_cols].values, y = df["salary_target"].values
2. train_test_split 80/20, random_state=42
3. Обучи: Ridge(alpha=1.0) в Pipeline со StandardScaler, RandomForestRegressor(n_estimators=50,random_state=42), GradientBoostingRegressor(n_estimators=50,random_state=42)
4. Метрики на test: MAE, RMSE, R²
5. Сохрани:
   - `best_model` — объект лучшей модели (по MAE)
   - `best_model_name` — строка
   - `model_results` — список [{"name","mae","rmse","r2"}]
   - `result` — {"status":"ok","best_model":str,"best_mae":float,"best_rmse":float,"best_r2":float}

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
Возвращай только Python-код без пояснений и без markdown-блоков.
""",
    ),
}


def get_llm(model_id: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


def clean_code(code: str) -> str:
    import re
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)
    code = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', code)
    return code


def run_prompt(llm, prompt: str, local_vars: dict, max_retries: int = 3):
    retries = 0
    response = llm.invoke(prompt)
    code = clean_code(response.content)
    original_vars = dict(local_vars)

    for attempt in range(max_retries):
        try:
            vars_copy = dict(original_vars)
            exec(compile(code, "<llm>", "exec"), vars_copy)
            local_vars.update(vars_copy)
            return code, retries, None
        except Exception as e:
            retries += 1
            err = traceback.format_exc()
            if attempt == max_retries - 1:
                return code, retries, str(e)
            fix_prompt = f"""Исправь Python-код. Верни только исправленный код без пояснений.

КОД:
{code}

ОШИБКА:
{err}
"""
            response = llm.invoke(fix_prompt)
            code = clean_code(response.content)

    return code, retries, "max_retries_exceeded"


def benchmark_model(model_info: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Тестирую: {model_info['name']} ({model_info['id']})")
    print('='*60)

    row = {
        "model_name": model_info["name"],
        "model_id": model_info["id"],
        "tier": model_info["tier"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        llm = get_llm(model_info["id"])
    except Exception as e:
        row["error"] = str(e)
        return row

    import pandas as pd
    import numpy as np

    # ---- EDA ----
    print("  [1/3] EDA...")
    t0 = time.time()
    _, key = "eda", PROMPTS["eda"]
    task_key, prompt = PROMPTS["eda"]
    local_vars = {}
    _, eda_retries, eda_error = run_prompt(llm, prompt, local_vars)
    eda_time = round(time.time() - t0, 2)
    eda_ok = eda_error is None and "result" in local_vars
    eda_rows = local_vars.get("result", {}).get("rows", None) if eda_ok else None

    row["eda_ok"] = int(eda_ok)
    row["eda_retries"] = eda_retries
    row["eda_time_sec"] = eda_time
    row["eda_rows_detected"] = eda_rows
    row["eda_error"] = eda_error or ""
    print(f"    {'OK' if eda_ok else 'FAIL'} — {eda_time}s, retries={eda_retries}")

    # ---- PREPROCESS ----
    print("  [2/3] Предобработка...")
    t0 = time.time()
    _, prompt = PROMPTS["preprocess"]
    local_vars = {}
    _, pre_retries, pre_error = run_prompt(llm, prompt, local_vars)
    pre_time = round(time.time() - t0, 2)
    pre_ok = pre_error is None and "df_processed" in local_vars and "feature_cols" in local_vars
    df_proc = local_vars.get("df_processed") if pre_ok else None
    pre_rows = len(df_proc) if df_proc is not None else None
    salary_mean = local_vars.get("result", {}).get("salary_mean", None) if pre_ok else None

    row["preprocess_ok"] = int(pre_ok)
    row["preprocess_retries"] = pre_retries
    row["preprocess_time_sec"] = pre_time
    row["preprocess_rows"] = pre_rows
    row["salary_mean"] = round(salary_mean, 0) if salary_mean else None
    row["preprocess_error"] = pre_error or ""
    print(f"    {'OK' if pre_ok else 'FAIL'} — {pre_time}s, retries={pre_retries}")

    # ---- TRAIN ----
    print("  [3/3] Обучение моделей...")
    t0 = time.time()
    _, prompt = PROMPTS["train"]
    if pre_ok and df_proc is not None:
        feature_cols = local_vars.get("feature_cols", [])
        train_vars = {"df": df_proc, "feature_cols": feature_cols}
    else:
        try:
            df_tmp = pd.read_csv(DATA_FILE)
            df_tmp["salary_target"] = df_tmp[["salary_from", "salary_to"]].mean(axis=1, skipna=True)
            df_tmp = df_tmp.dropna(subset=["salary_target"])
            exp_map = {"noExperience": 0, "between1And3": 2, "between3And6": 4, "moreThan6": 7}
            df_tmp["experience_years"] = df_tmp["experience"].map(exp_map).fillna(3)
            df_tmp["is_moscow"] = (df_tmp["city"] == "Москва").astype(int)
            df_tmp["is_spb"] = (df_tmp["city"] == "Санкт-Петербург").astype(int)
            df_tmp["skills_count"] = df_tmp["skills"].str.split(";").apply(len)
            df_tmp["skill_python"] = df_tmp["skills"].str.contains("Python", case=False, na=False).astype(int)
            feature_cols = ["experience_years", "is_moscow", "is_spb", "skills_count", "skill_python"]
            train_vars = {"df": df_tmp, "feature_cols": feature_cols}
        except Exception:
            train_vars = {}

    _, train_retries, train_error = run_prompt(llm, prompt, train_vars)
    train_time = round(time.time() - t0, 2)
    train_ok = train_error is None and "result" in train_vars

    result = train_vars.get("result", {}) if train_ok else {}
    row["train_ok"] = int(train_ok)
    row["train_retries"] = train_retries
    row["train_time_sec"] = train_time
    row["best_model"] = result.get("best_model", "")
    row["best_mae"] = round(result.get("best_mae", 0), 0) if train_ok else None
    row["best_rmse"] = round(result.get("best_rmse", 0), 0) if train_ok else None
    row["best_r2"] = round(result.get("best_r2", 0), 4) if train_ok else None
    row["train_error"] = train_error or ""
    print(f"    {'OK' if train_ok else 'FAIL'} — {train_time}s, retries={train_retries}")

    # ---- ИТОГО ----
    row["total_retries"] = eda_retries + pre_retries + train_retries
    row["total_time_sec"] = round(eda_time + pre_time + train_time, 2)
    row["tasks_passed"] = int(eda_ok) + int(pre_ok) + int(train_ok)
    row["success_rate_pct"] = round(row["tasks_passed"] / 3 * 100, 1)

    return row


FIELDNAMES = [
    "model_name", "model_id", "tier", "timestamp",
    "eda_ok", "eda_retries", "eda_time_sec", "eda_rows_detected", "eda_error",
    "preprocess_ok", "preprocess_retries", "preprocess_time_sec", "preprocess_rows", "salary_mean", "preprocess_error",
    "train_ok", "train_retries", "train_time_sec", "best_model", "best_mae", "best_rmse", "best_r2", "train_error",
    "total_retries", "total_time_sec", "tasks_passed", "success_rate_pct",
    "error",
]

OUTPUT_FILE = "llm_comparison.csv"


def main():
    print(f"Бенчмарк LLM-моделей — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Тестируем {len(MODELS)} моделей на {DATA_FILE}\n")

    results = []
    for model_info in MODELS:
        try:
            row = benchmark_model(model_info)
        except Exception as e:
            row = {
                "model_name": model_info["name"],
                "model_id": model_info["id"],
                "tier": model_info["tier"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "tasks_passed": 0,
                "success_rate_pct": 0,
            }
        results.append(row)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Результаты сохранены в {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"{'Модель':<22} {'Tier':<6} {'Tasks':<8} {'MAE':>10} {'R2':>7} {'Time':>8} {'Retries':>8}")
    print("-" * 70)
    for row in results:
        print(
            f"{row.get('model_name','?'):<22} "
            f"{row.get('tier','?'):<6} "
            f"{str(row.get('tasks_passed','?'))+'/3':<8} "
            f"{str(row.get('best_mae') or '-'):>10} "
            f"{str(row.get('best_r2') or '-'):>7} "
            f"{str(row.get('total_time_sec','?'))+'s':>8} "
            f"{str(row.get('total_retries','?')):>8}"
        )


if __name__ == "__main__":
    main()
