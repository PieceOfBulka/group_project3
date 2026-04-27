"""
tools/tool_predict.py
Tool 4: Предсказание зарплаты для конкретной вакансии.
"""

import json
import traceback
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.predict")


@tool
def predict_salary(vacancy_json: str) -> str:
    """
    Предсказывает зарплату для вакансии без указанной зарплаты.
    Вызывай после train_and_compare_models.

    Аргумент vacancy_json: JSON-строка с полями вакансии, например:
    {"name": "Senior Python Developer", "experience": "between3And6",
     "employment": "full", "schedule": "remote", "city": "Москва",
     "skills": "Python;FastAPI;Docker;PostgreSQL"}

    Поля experience: noExperience | between1And3 | between3And6 | moreThan6
    """
    logger.info(f"Предсказание зарплаты | vacancy={vacancy_json[:80]}")
    if STATE["best_model"] is None:
        logger.error("best_model не найден в STATE")
        return json.dumps({"status": "error", "message": "Сначала вызови train_and_compare_models"})

    model = STATE["best_model"]
    model_name = STATE["best_model_name"]
    feature_cols = STATE["feature_cols"]

    llm = get_llm()

    prompt = f"""Ты — ML-инженер. Напиши Python-код для преобразования вакансии в вектор признаков.

ВАКАНСИЯ (уже доступна как переменная `vacancy` — dict):
{vacancy_json}

СПИСОК ПРИЗНАКОВ (уже доступен как переменная `feature_cols` — list):
{json.dumps(feature_cols)}

ЗАДАЧА:
Создай pandas DataFrame с ОДНОЙ строкой, колонки = feature_cols.

ПРАВИЛА:
- experience_years: noExperience→0, between1And3→2, between3And6→4, moreThan6→7, иначе→3
- employment_score: full→1.0, part→0.5, project→0.5, иначе→1.0
- schedule_score: fullDay→1.0, remote→0.7, flexible→0.8, иначе→1.0
- is_moscow: 1 если vacancy["city"]=="Москва"
- is_spb: 1 если vacancy["city"]=="Санкт-Петербург"
- is_top_city: 1 если city в [Москва, Санкт-Петербург, Новосибирск, Екатеринбург, Казань]
- is_senior: 1 если senior/lead/старший/руководитель в vacancy["name"] (lower)
- is_junior: 1 если junior/intern/стажёр/младший в vacancy["name"] (lower)
- skills_count: кол-во навыков (разделитель ";" или ",")
- skill_python, skill_sql и т.д.: 1 если навык есть в vacancy["skills"]

СОХРАНИ:
- `features_df` — DataFrame одна строка, колонки строго из feature_cols
- `result` — {{"status": "ok"}}

import pandas as pd
Все отсутствующие признаки = 0.
Возвращай только Python-код без пояснений и без markdown-блоков.
"""

    response = llm.invoke(prompt)

    try:
        vacancy = json.loads(vacancy_json)
        local_vars = {"vacancy": vacancy, "feature_cols": feature_cols}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        features_df = local_vars["features_df"]
        prediction = model.predict(features_df[feature_cols].values)[0]
        rounded = round(prediction / 5000) * 5000

        log_action("predict_salary", f"{vacancy.get('name','?')} → {rounded:,.0f} ₽")
        logger.info(f"Предсказание: {vacancy.get('name','?')} → {rounded:,.0f} ₽ (модель: {model_name})")
        return json.dumps({
            "status": "ok",
            "vacancy_name": vacancy.get("name", "Вакансия"),
            "model_used": model_name,
            "predicted_salary_rur": rounded,
            "predicted_salary_formatted": f"{rounded:,.0f} ₽".replace(",", " "),
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Предсказание ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})