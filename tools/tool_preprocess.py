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
from tools.logger import get_logger

logger = get_logger("tool.preprocess")


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
КОЛОНКИ: name, experience, employment, schedule, city, company, description, skills, category, level, salary_from, salary_to, salary

ЗАДАЧА — выполни следующие шаги:

1. ЗАГРУЗКА:
   - df = pd.read_csv("{filepath}")

2. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (salary_target):
   - Если есть колонка "salary" и значение не NaN — используй её
   - Иначе: среднее salary_from и salary_to (skipna=True)
   - Удали строки где salary_target NaN
   - Сохрани в колонку "salary_target"

3. ПРИЗНАК ИЗ ОПЫТА:
   - "noExperience"→0, "between1And3"→2, "between3And6"→4, "moreThan6"→7, иначе→3
   - Колонка: "experience_years"

4. ПРИЗНАКИ ИЗ ГОРОДА:
   - is_moscow = (city == "Москва").astype(int)
   - is_spb = (city == "Санкт-Петербург").astype(int)
   - is_remote = (city.str.lower().str.contains("удалённо|удаленно", na=False)).astype(int)
   - is_top_city: 1 если город в [Москва, Санкт-Петербург, Новосибирск, Екатеринбург, Казань]

5. ЗАНЯТОСТЬ И ГРАФИК:
   - employment_score: full→1.0, part→0.5, project→0.5, иначе→1.0
   - schedule_score: fullDay→1.0, remote→0.7, flexible→0.8, иначе→1.0

6. УРОВЕНЬ ПОЗИЦИИ:
   - Из колонки "level" если есть: junior→0, middle→1, senior→2, lead→3, иначе→1
   - Колонка: level_score
   - is_senior: 1 если level в [senior, lead] или "senior"/"lead" в name (lower)
   - is_junior: 1 если level == junior или "junior"/"intern" в name (lower)

7. КАТЕГОРИЯ (из колонки category если есть):
   - Закодируй через pd.get_dummies с prefix="cat", drop_first=True

8. ПРИЗНАКИ ИЗ НАВЫКОВ:
   - Разделитель ";" или ","
   - skills_count: количество навыков
   - Бинарные: skill_python, skill_sql, skill_docker, skill_kubernetes,
     skill_pytorch, skill_tensorflow, skill_javascript, skill_react,
     skill_java, skill_go, skill_spark, skill_airflow, skill_postgresql,
     skill_redis, skill_kafka, skill_git, skill_linux, skill_pandas

9. РЕЗУЛЬТАТ:
   - `df_processed` — итоговый DataFrame
   - `feature_cols` — список признаков (исключи: salary_target, salary_from, salary_to,
     salary, id, url, description, name, company, city, experience, employment,
     schedule, skills, salary_currency, category, level)
   - `result` — {{"status": "ok", "rows": int, "feature_cols": list,
       "salary_mean": float, "salary_min": float, "salary_max": float}}

ВАЖНО:
- import pandas as pd, import numpy as np
- Все признаки заполняй 0 если исходной колонки нет
- Возвращай только Python-код без пояснений и без markdown-блоков
"""

    logger.info(f"Предобработка | filepath={filepath}")
    response = llm.invoke(prompt)

    try:
        local_vars = {}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        STATE["df_processed"] = local_vars["df_processed"]
        STATE["feature_cols"] = local_vars["feature_cols"]

        result = local_vars.get("result", {})
        result["feature_cols"] = local_vars["feature_cols"]
        log_action("preprocess_data", f"Обработано {result.get('rows','?')} строк, {len(local_vars['feature_cols'])} признаков")
        logger.info(f"Предобработка завершена | rows={result.get('rows','?')} | features={len(local_vars['feature_cols'])} | salary_mean={result.get('salary_mean','?'):.0f}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Предобработка ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})