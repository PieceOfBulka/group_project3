"""
tools/tool_preprocess.py
Tool 2: Предобработка данных — целевая переменная, признаки, кодирование.
"""

import json
import traceback
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE, log_action


@tool
def preprocess_data(filepath: str) -> str:
    """
    Предобрабатывает CSV с вакансиями HH.ru: вычисляет целевую переменную (зарплату),
    кодирует категории, создаёт признаки из навыков.
    Аргумент: filepath — путь к CSV файлу.
    """
    llm = get_llm()

    prompt = f"""Ты — опытный ML-инженер. Напиши Python-код для предобработки датасета вакансий с HH.ru.

ФАЙЛ: {filepath}

ЗАДАЧА — выполни следующие шаги:

1. ЗАГРУЗКА:
   - Загрузи CSV с помощью pandas

2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (salary_target):
   - Найди колонки с зарплатой (обычно salary_from, salary_to или salary)
   - Если есть salary_from и salary_to — вычисли среднее (skipna=True)
   - Если валюта USD — умножь на 90, EUR — на 100
   - Удали строки где salary_target is NaN
   - Сохрани в колонку "salary_target"

3. ПРИЗНАК ИЗ ОПЫТА:
   - "noExperience"→0, "between1And3"→2, "between3And6"→4, "moreThan6"→7, иначе→3
   - Колонка: "experience_years"

4. ПРИЗНАКИ ИЗ ГОРОДА:
   - is_moscow, is_spb, is_top_city (топ: Москва, Санкт-Петербург, Новосибирск, Екатеринбург, Казань)

5. ЗАНЯТОСТЬ И ГРАФИК:
   - employment_score: full→1.0, part→0.5, project→0.5, иначе→1.0
   - schedule_score: fullDay→1.0, remote→0.7, flexible→0.8, иначе→1.0

6. УРОВЕНЬ ПОЗИЦИИ (из колонки name):
   - is_senior: senior/lead/старший/руководитель в названии (case-insensitive)
   - is_junior: junior/intern/стажёр/младший в названии (case-insensitive)

7. ПРИЗНАКИ ИЗ НАВЫКОВ:
   - Разделитель может быть "," или ";"
   - Бинарные колонки для: Python, SQL, Docker, Kubernetes, pandas, sklearn,
     PyTorch, TensorFlow, JavaScript, React, Java, Go, Spark, Airflow,
     PostgreSQL, Redis, Kafka, Git
   - Имена колонок: skill_python, skill_sql, skill_docker и т.д. (lower)
   - skills_count: общее количество навыков

8. РЕЗУЛЬТАТ:
   - `df_processed` — итоговый DataFrame
   - `feature_cols` — список признаков (исключи: salary_target, salary_from, salary_to,
     salary, id, url, description, name, company, city, experience, employment,
     schedule, skills, salary_currency)
   - `result` — словарь:
     {{"status": "ok", "rows": int, "feature_cols": list,
       "salary_mean": float, "salary_min": float, "salary_max": float}}

ВАЖНО:
- import pandas as pd, import numpy as np
- Все признаки-колонки заполняй 0 если исходной колонки нет в данных
- Возвращай только Python-код без пояснений и без markdown-блоков
"""

    response = llm.invoke(prompt)

    try:
        local_vars = {}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        STATE["df_processed"] = local_vars["df_processed"]
        STATE["feature_cols"] = local_vars["feature_cols"]

        result = local_vars.get("result", {})
        result["feature_cols"] = local_vars["feature_cols"]
        log_action("preprocess_data", f"Обработано {result.get('rows','?')} строк, {len(local_vars['feature_cols'])} признаков")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})