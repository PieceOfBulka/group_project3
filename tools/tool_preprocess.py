"""
tools/tool_preprocess.py
Tool 2: Предобработка данных — агент сам пишет код через LLM.
"""

import json
import traceback
import pandas as pd
import numpy as np
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
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
    logger.info(f"Preprocess | filepath={filepath}")
    llm = get_llm()

    # Получаем df_raw из STATE или читаем заново
    df_raw = STATE.get("df_raw")
    if df_raw is None:
        try:
            df_raw = pd.read_csv(filepath)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Не удалось загрузить файл: {e}"})

    columns_info = list(df_raw.columns)
    dtypes_info = df_raw.dtypes.astype(str).to_dict()
    sample_info = df_raw.head(2).to_dict("records")
    shape_info = df_raw.shape

    prompt = f"""Ты — опытный ML-инженер. Напиши Python-код для предобработки датасета вакансий с HH.ru.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже готовы, не нужно импортировать или переопределять):
- df — pandas DataFrame с исходными данными ({shape_info[0]} строк, {shape_info[1]} колонок)
- pd — pandas
- np — numpy

МЕТА-ИНФОРМАЦИЯ О ДАННЫХ:
- Колонки: {columns_info}
- Типы: {dtypes_info}
- Примеры строк: {sample_info}

ЗАДАЧА — выполни шаги строго по порядку:

1. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (salary_target):
   - Найди колонки с зарплатой среди: salary_from, salary_to, salary (ищи по наличию в df.columns)
   - Если есть salary_from и salary_to → salary_target = среднее(salary_from, salary_to) по строке (skipna=True)
   - Если есть только salary_from → salary_target = salary_from
   - Если есть только salary_to → salary_target = salary_to
   - Если есть salary → salary_target = salary
   - Если есть колонка salary_currency или currency — для USD умножь на 90, для EUR на 100
   - Удали строки где salary_target is NaN
   - Сохрани результат в df["salary_target"]

2. КОДИРОВАНИЕ ОПЫТА (experience_years):
   - Если в df есть колонка experience: "noExperience"→0, "between1And3"→2, "between3And6"→4, "moreThan6"→7, иначе→3
   - Если колонки нет — df["experience_years"] = 3

3. ПРИЗНАКИ ГОРОДА:
   - is_moscow: 1 если city == "Москва" (case-sensitive), иначе 0
   - is_spb: 1 если city == "Санкт-Петербург", иначе 0
   - is_top_city: 1 если city в ["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань"], иначе 0
   - Если колонки city нет — все три = 0

4. ЗАНЯТОСТЬ И ГРАФИК:
   - employment_score: full→1.0, part→0.5, project→0.5, иначе→1.0 (по колонке employment)
   - schedule_score: fullDay→1.0, remote→0.7, flexible→0.8, иначе→1.0 (по колонке schedule)
   - Если колонок нет — значение по умолчанию 1.0

5. УРОВЕНЬ ПОЗИЦИИ (по колонке name):
   - is_senior: 1 если в lower(name) есть "senior","lead","старший","руководитель" иначе 0
   - is_junior: 1 если в lower(name) есть "junior","intern","стажёр","младший" иначе 0
   - Если колонки name нет — оба 0

6. ПРИЗНАКИ НАВЫКОВ (по колонке skills):
   - Разделитель: ";" или "," — определи по первой непустой строке
   - Для каждого навыка из списка ["Python","SQL","Docker","Kubernetes","pandas","sklearn",
     "PyTorch","TensorFlow","JavaScript","React","Java","Go","Spark","Airflow",
     "PostgreSQL","Redis","Kafka","Git"] — создай бинарную колонку skill_<lower>
     (например skill_python, skill_sql, skill_docker и т.д.)
   - 1 если навык встречается в строке навыков (case-insensitive), иначе 0
   - skills_count: количество навыков в строке (0 если пусто)
   - Если колонки skills нет — все skill_* = 0, skills_count = 0

7. СПИСОК ПРИЗНАКОВ (feature_cols):
   - Включи все новые колонки: experience_years, is_moscow, is_spb, is_top_city,
     employment_score, schedule_score, is_senior, is_junior, skills_count,
     skill_python, skill_sql, skill_docker, skill_kubernetes, skill_pandas,
     skill_sklearn, skill_pytorch, skill_tensorflow, skill_javascript, skill_react,
     skill_java, skill_go, skill_spark, skill_airflow, skill_postgresql, skill_redis,
     skill_kafka, skill_git
   - Оставь только те из них, которые реально есть в df.columns
   - feature_cols = [col for col in desired_cols if col in df.columns]

8. ФИНАЛЬНАЯ ОЧИСТКА:
   - Заполни NaN в feature_cols нулями: df[feature_cols] = df[feature_cols].fillna(0)
   - df_processed = df (итоговый датафрейм)

9. РЕЗУЛЬТАТ:
   - result = {{
       "status": "ok",
       "rows": len(df_processed),
       "feature_cols": feature_cols,
       "salary_mean": float(df_processed["salary_target"].mean()),
       "salary_min": float(df_processed["salary_target"].min()),
       "salary_max": float(df_processed["salary_target"].max()),
     }}

--- FEW-SHOT ПРИМЕР (правильный код) ---
import pandas as pd
import numpy as np

# Шаг 1: salary_target
if "salary_from" in df.columns and "salary_to" in df.columns:
    df["salary_target"] = df[["salary_from", "salary_to"]].mean(axis=1, skipna=True)
elif "salary_from" in df.columns:
    df["salary_target"] = df["salary_from"]
elif "salary" in df.columns:
    df["salary_target"] = df["salary"]
df = df.dropna(subset=["salary_target"])

# Шаг 2: experience_years
exp_map = {{"noExperience": 0, "between1And3": 2, "between3And6": 4, "moreThan6": 7}}
df["experience_years"] = df["experience"].map(exp_map).fillna(3) if "experience" in df.columns else 3

# Шаг 3: city flags
top_cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань"]
if "city" in df.columns:
    df["is_moscow"] = (df["city"] == "Москва").astype(int)
    df["is_spb"] = (df["city"] == "Санкт-Петербург").astype(int)
    df["is_top_city"] = df["city"].isin(top_cities).astype(int)
else:
    df["is_moscow"] = df["is_spb"] = df["is_top_city"] = 0

# и так далее...

desired_cols = ["experience_years","is_moscow","is_spb","is_top_city","employment_score",
                "schedule_score","is_senior","is_junior","skills_count",
                "skill_python","skill_sql","skill_docker","skill_kubernetes","skill_pandas",
                "skill_sklearn","skill_pytorch","skill_tensorflow","skill_javascript","skill_react",
                "skill_java","skill_go","skill_spark","skill_airflow","skill_postgresql",
                "skill_redis","skill_kafka","skill_git"]
feature_cols = [c for c in desired_cols if c in df.columns]
df_processed = df
result = {{"status":"ok","rows":len(df_processed),"feature_cols":feature_cols,
           "salary_mean":float(df_processed["salary_target"].mean()),
           "salary_min":float(df_processed["salary_target"].min()),
           "salary_max":float(df_processed["salary_target"].max())}}

--- АНТИПРИМЕР (не делай так) ---
# df["salary_target"] = df.salary_from + df.salary_to  # ОШИБКА: нет проверки на существование колонок
# feature_cols = ["experience_years", ...]  # ОШИБКА: хардкодить без проверки if col in df.columns
# df_processed = df.copy()  # допустимо, но не обязательно

ТРЕБОВАНИЯ:
- Используй ТОЛЬКО переменные df, pd, np (уже доступны, не объявляй их заново)
- Все признаки заполни 0 если исходная колонка отсутствует
- Не используй markdown-форматирование
- Верни ТОЛЬКО Python-код, без пояснений
"""

    response = llm.invoke(prompt)
    logger.info("LLM сгенерировал код предобработки")

    try:
        local_vars = {"pd": pd, "np": np, "df": df_raw.copy()}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        df_processed = local_vars.get("df_processed")
        feature_cols = local_vars.get("feature_cols", [])

        if df_processed is None:
            raise ValueError("LLM-код не создал переменную df_processed")

        STATE["df_processed"] = df_processed
        STATE["feature_cols"] = feature_cols

        result = local_vars.get("result", {})
        result["feature_cols"] = feature_cols
        log_action("preprocess_data", f"Обработано {result.get('rows','?')} строк, {len(feature_cols)} признаков")
        logger.info(f"Preprocess завершён | rows={result.get('rows')} | features={len(feature_cols)}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Preprocess ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
