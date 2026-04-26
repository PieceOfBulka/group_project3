"""
tools/tool_train.py
Tool 3: Обучение трёх моделей, сравнение, выбор лучшей.
"""

import json
import traceback
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE


@tool
def train_and_compare_models(dummy: str = "") -> str:
    """
    Обучает три ML модели (LinearRegression, RandomForest, GradientBoosting),
    сравнивает их по MAE/RMSE/R² и выбирает лучшую.
    Вызывай после preprocess_data. Аргумент dummy можно передать пустым "".
    """
    if STATE["df_processed"] is None:
        return json.dumps({"status": "error", "message": "Сначала вызови preprocess_data"})

    df = STATE["df_processed"]
    feature_cols = STATE["feature_cols"]

    context = {
        "rows": len(df),
        "feature_cols": feature_cols,
        "salary_mean": float(df["salary_target"].mean()),
        "salary_std": float(df["salary_target"].std()),
    }

    llm = get_llm()

    prompt = f"""Ты — ML-инженер. Напиши Python-код для обучения и сравнения трёх моделей регрессии.

КОНТЕКСТ ДАННЫХ:
{json.dumps(context, ensure_ascii=False)}

Переменные уже доступны в коде:
- `df` — pandas DataFrame с колонками salary_target и признаками из feature_cols
- `feature_cols` — список названий признаков (list)

ЗАДАЧА:

1. ПОДГОТОВКА:
   - X = df[feature_cols].values
   - y = df["salary_target"].values
   - train_test_split 80/20, random_state=42

2. ТРИ МОДЕЛИ:
   a) LinearRegression → Ridge(alpha=1.0) в Pipeline со StandardScaler
   b) RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
   c) GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

3. МЕТРИКИ на test: MAE, RMSE, R²

4. СОХРАНИ:
   - `best_model` — объект лучшей модели (по наименьшему MAE)
   - `best_model_name` — строка с именем
   - `model_results` — список dict [{{"name", "mae", "rmse", "r2"}}], по MAE ascending
   - `result` — {{"status":"ok", "best_model":str, "best_mae":float,
                  "best_rmse":float, "best_r2":float,
                  "all_models":model_results, "train_size":int, "test_size":int}}

ИМПОРТЫ:
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

Возвращай только Python-код без пояснений и без markdown-блоков.
"""

    response = llm.invoke(prompt)

    try:
        local_vars = {"df": df, "feature_cols": feature_cols}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        STATE["best_model"] = local_vars["best_model"]
        STATE["best_model_name"] = local_vars["best_model_name"]
        STATE["model_results"] = local_vars["model_results"]

        return json.dumps(local_vars.get("result", {}), ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})