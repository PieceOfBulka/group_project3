"""
tools/tool_predict.py
Tool 4: Предсказание зарплаты — агент сам пишет код через LLM.
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
    if STATE.get("best_model") is None:
        return json.dumps({"status": "error", "message": "Сначала вызови train_and_compare_models"})

    model = STATE["best_model"]
    model_name = STATE["best_model_name"]
    feature_cols = STATE["feature_cols"]

    llm = get_llm()

    prompt = f"""Ты — ML-инженер. Напиши Python-код для преобразования описания вакансии в вектор признаков для ML-модели.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже готовы, не переопределяй):
- vacancy — dict с описанием вакансии
- feature_cols — список признаков модели
- pd — pandas

ВАКАНСИЯ:
{vacancy_json}

СПИСОК ПРИЗНАКОВ МОДЕЛИ:
{json.dumps(feature_cols)}

ЗАДАЧА — создай pandas DataFrame с одной строкой, где колонки = feature_cols.
Каждый признак вычисли по правилам:

1. experience_years:
   vacancy["experience"]: "noExperience"→0, "between1And3"→2, "between3And6"→4, "moreThan6"→7, иначе→3

2. employment_score:
   vacancy["employment"]: "full"→1.0, "part"→0.5, "project"→0.5, иначе→1.0

3. schedule_score:
   vacancy["schedule"]: "fullDay"→1.0, "remote"→0.7, "flexible"→0.8, иначе→1.0

4. Флаги города (из vacancy["city"]):
   - is_moscow: 1 если city == "Москва"
   - is_spb: 1 если city == "Санкт-Петербург"
   - is_top_city: 1 если city в ["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань"]

5. Флаги позиции (из lower(vacancy["name"])):
   - is_senior: 1 если есть "senior","lead","старший","руководитель"
   - is_junior: 1 если есть "junior","intern","стажёр","младший"

6. Навыки (из vacancy["skills"]):
   - Разделитель ";" или "," — определи автоматически
   - skills_count: количество навыков
   - Для каждой skill_* колонки из feature_cols: 1 если навык присутствует (case-insensitive)
     Маппинг: skill_python→"python", skill_sql→"sql", skill_docker→"docker",
     skill_kubernetes→"kubernetes", skill_pandas→"pandas", skill_sklearn→"sklearn",
     skill_pytorch→"pytorch", skill_tensorflow→"tensorflow", skill_javascript→"javascript",
     skill_react→"react", skill_java→"java", skill_go→"go", skill_spark→"spark",
     skill_airflow→"airflow", skill_postgresql→"postgresql", skill_redis→"redis",
     skill_kafka→"kafka", skill_git→"git"

7. ИТОГ:
   - row = {{col: 0 for col in feature_cols}}  # начало с нулей
   - Заполни вычисленные значения
   - features_df = pd.DataFrame([row])[feature_cols]
   - result = {{"status": "ok"}}

--- FEW-SHOT ПРИМЕР ---
import pandas as pd

exp_map = {{"noExperience": 0, "between1And3": 2, "between3And6": 4, "moreThan6": 7}}
skills_str = str(vacancy.get("skills", ""))
sep = ";" if ";" in skills_str else ","
skills_list = [s.strip().lower() for s in skills_str.split(sep) if s.strip()]

row = {{col: 0 for col in feature_cols}}
if "experience_years" in row:
    row["experience_years"] = exp_map.get(vacancy.get("experience", ""), 3)
if "employment_score" in row:
    row["employment_score"] = {{"full":1.0,"part":0.5,"project":0.5}}.get(vacancy.get("employment",""), 1.0)
if "schedule_score" in row:
    row["schedule_score"] = {{"fullDay":1.0,"remote":0.7,"flexible":0.8}}.get(vacancy.get("schedule",""), 1.0)
city = vacancy.get("city","")
if "is_moscow" in row: row["is_moscow"] = int(city == "Москва")
if "is_spb" in row: row["is_spb"] = int(city == "Санкт-Петербург")
if "is_top_city" in row: row["is_top_city"] = int(city in ["Москва","Санкт-Петербург","Новосибирск","Екатеринбург","Казань"])
name_lower = vacancy.get("name","").lower()
if "is_senior" in row: row["is_senior"] = int(any(k in name_lower for k in ["senior","lead","старший","руководитель"]))
if "is_junior" in row: row["is_junior"] = int(any(k in name_lower for k in ["junior","intern","стажёр","младший"]))
if "skills_count" in row: row["skills_count"] = len(skills_list)
skill_map = {{"skill_python":"python","skill_sql":"sql","skill_docker":"docker","skill_kubernetes":"kubernetes",
              "skill_pandas":"pandas","skill_sklearn":"sklearn","skill_pytorch":"pytorch",
              "skill_tensorflow":"tensorflow","skill_javascript":"javascript","skill_react":"react",
              "skill_java":"java","skill_go":"go","skill_spark":"spark","skill_airflow":"airflow",
              "skill_postgresql":"postgresql","skill_redis":"redis","skill_kafka":"kafka","skill_git":"git"}}
for col, kw in skill_map.items():
    if col in row: row[col] = int(kw in skills_list)

features_df = pd.DataFrame([row])[feature_cols]
result = {{"status": "ok"}}

--- АНТИПРИМЕР ---
# features_df = pd.DataFrame()  # ОШИБКА: пустой DataFrame — модель не сможет предсказать
# row["skill_python"] = 1  # ОШИБКА: без проверки if "skill_python" in row

ТРЕБОВАНИЯ:
- Используй только vacancy, feature_cols, pd (уже доступны)
- features_df должен иметь СТРОГО колонки из feature_cols, в том же порядке
- Все отсутствующие признаки = 0
- Верни ТОЛЬКО Python-код, без пояснений и без markdown
"""

    response = llm.invoke(prompt)
    logger.info("LLM сгенерировал код построения вектора признаков")

    try:
        import pandas as pd
        vacancy = json.loads(vacancy_json)
        local_vars = {"vacancy": vacancy, "feature_cols": feature_cols, "pd": pd}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        features_df = local_vars.get("features_df")
        if features_df is None:
            raise ValueError("LLM-код не создал переменную features_df")

        prediction = model.predict(features_df[feature_cols].values)[0]
        rounded = round(prediction / 5000) * 5000

        log_action("predict_salary", f"{vacancy.get('name','?')} → {rounded:,.0f} руб")
        logger.info(f"Predict завершён | {vacancy.get('name')} → {rounded:,.0f}")
        return json.dumps({
            "status": "ok",
            "vacancy_name": vacancy.get("name", "Вакансия"),
            "model_used": model_name,
            "predicted_salary_rur": rounded,
            "predicted_salary_formatted": f"{rounded:,.0f} руб".replace(",", " "),
            "features_used": len(feature_cols),
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Predict ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
