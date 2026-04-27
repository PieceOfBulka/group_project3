"""
tools/tool_train.py
Tool 3: Обучение трёх моделей, сравнение, выбор лучшей.
"""

import json
import traceback
from langchain_core.tools import tool

import os
import pickle

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.train")

MEMORY_DIR = "memory"


@tool
def train_and_compare_models(dummy: str = "") -> str:
    """
    Обучает три ML модели (LinearRegression, RandomForest, GradientBoosting),
    сравнивает их по MAE/RMSE/R² и выбирает лучшую.
    Вызывай после preprocess_data. Аргумент dummy можно передать пустым "".
    """
    logger.info("Начало обучения моделей")
    if STATE["df_processed"] is None:
        logger.error("df_processed не найден в STATE — preprocess_data не был вызван")
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

        result = local_vars.get("result", {})

        os.makedirs(MEMORY_DIR, exist_ok=True)
        model_path = os.path.join(MEMORY_DIR, "best_model.pkl")
        prev_mae = float("inf")
        if os.path.exists(model_path):
            try:
                with open(os.path.join(MEMORY_DIR, "best_metrics.json")) as f:
                    import json as _json
                    prev = _json.load(f)
                    prev_mae = prev.get("best_mae", float("inf"))
            except Exception:
                pass

        current_mae = result.get("best_mae", float("inf"))
        if current_mae < prev_mae:
            with open(model_path, "wb") as f:
                pickle.dump(local_vars["best_model"], f)
            result["model_saved"] = True
            result["previous_mae"] = prev_mae
            result["improvement"] = prev_mae - current_mae
        else:
            result["model_saved"] = False
            result["previous_mae"] = prev_mae

        log_action("train_and_compare_models",
                   f"Лучшая: {local_vars['best_model_name']}, MAE={current_mae:,.0f}")
        logger.info(f"Обучение завершено | best={local_vars['best_model_name']} | MAE={current_mae:,.0f} | R2={result.get('best_r2',0):.4f} | saved={result.get('model_saved')}")
        for m in local_vars.get("model_results", []):
            logger.info(f"  Модель: {m['name']} | MAE={m['mae']:,.0f} | RMSE={m['rmse']:,.0f} | R2={m['r2']:.4f}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Обучение ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})