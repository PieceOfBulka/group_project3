"""
tools/tool_train.py
Tool 3: Обучение трёх моделей — агент сам пишет код через LLM.
"""

import json
import os
import pickle
import traceback
import numpy as np
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.train")

MEMORY_DIR = "memory"


@tool
def train_and_compare_models(dummy: str = "") -> str:
    """
    Обучает три ML модели (Ridge, RandomForest, GradientBoosting),
    сравнивает их по MAE/RMSE/R² и выбирает лучшую.
    Вызывай после preprocess_data. Аргумент dummy можно передать пустым "".
    """
    if STATE.get("df_processed") is None:
        return json.dumps({"status": "error", "message": "Сначала вызови preprocess_data"})

    df = STATE["df_processed"]
    feature_cols = STATE["feature_cols"]

    context = {
        "rows": len(df),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "salary_mean": float(df["salary_target"].mean()),
        "salary_std": float(df["salary_target"].std()),
        "salary_min": float(df["salary_target"].min()),
        "salary_max": float(df["salary_target"].max()),
    }

    llm = get_llm()

    prompt = f"""Ты — ML-инженер. Напиши Python-код для обучения, подбора гиперпараметров и сравнения трёх моделей регрессии.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже готовы, не переопределяй):
- df — pandas DataFrame с колонками salary_target и признаками
- feature_cols — список признаков: {json.dumps(feature_cols)}

КОНТЕКСТ ДАННЫХ:
{json.dumps(context, ensure_ascii=False, indent=2)}

ЗАДАЧА — выполни шаги строго по порядку:

1. ПОДГОТОВКА ДАННЫХ:
   - X = df[feature_cols].fillna(0).values
   - y = df["salary_target"].values
   - train_test_split: test_size=0.2, random_state=42

2. МОДЕЛЬ A — Ridge в Pipeline:
   - Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
   - GridSearchCV с param_grid = {{"model__alpha": [0.1, 1.0, 10.0, 100.0]}}
   - cv=3, scoring="neg_mean_absolute_error"
   - Обучи на X_train, y_train
   - Сохрани лучшую модель как model_ridge

3. МОДЕЛЬ B — RandomForestRegressor:
   - GridSearchCV с param_grid = {{"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]}}
   - cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
   - Сохрани лучшую модель как model_rf

4. МОДЕЛЬ C — GradientBoostingRegressor:
   - GridSearchCV с param_grid = {{"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 4, 5]}}
   - cv=3, scoring="neg_mean_absolute_error"
   - Сохрани лучшую модель как model_gb

5. ВЫЧИСЛЕНИЕ МЕТРИК на X_test, y_test для каждой модели:
   - MAE = mean_absolute_error(y_test, y_pred)
   - RMSE = sqrt(mean_squared_error(y_test, y_pred))
   - R2 = r2_score(y_test, y_pred)

6. ВЫБОР ЛУЧШЕЙ МОДЕЛИ:
   - model_results = список dict [{{"name": str, "mae": float, "rmse": float, "r2": float}}]
   - Отсортируй по mae ascending
   - best_model_name = имя модели с наименьшим MAE
   - best_model = соответствующий объект модели

7. РЕЗУЛЬТАТ:
   - result = {{
       "status": "ok",
       "best_model": best_model_name,
       "best_mae": float(лучший MAE),
       "best_rmse": float(лучший RMSE),
       "best_r2": float(лучший R2),
       "all_models": model_results,
       "train_size": len(X_train),
       "test_size": len(X_test),
     }}

--- FEW-SHOT ПРИМЕР (правильный код) ---
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df[feature_cols].fillna(0).values
y = df["salary_target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge
pipe_ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
gs_ridge = GridSearchCV(pipe_ridge, {{"model__alpha": [0.1, 1.0, 10.0, 100.0]}}, cv=3, scoring="neg_mean_absolute_error")
gs_ridge.fit(X_train, y_train)
model_ridge = gs_ridge.best_estimator_

# RandomForest
gs_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                     {{"n_estimators": [50, 100], "max_depth": [5, 10]}}, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
gs_rf.fit(X_train, y_train)
model_rf = gs_rf.best_estimator_

# GradientBoosting
gs_gb = GridSearchCV(GradientBoostingRegressor(random_state=42),
                     {{"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]}},
                     cv=3, scoring="neg_mean_absolute_error")
gs_gb.fit(X_train, y_train)
model_gb = gs_gb.best_estimator_

models = {{"Ridge": model_ridge, "RandomForest": model_rf, "GradientBoosting": model_gb}}
model_results = []
for name, mdl in models.items():
    pred = mdl.predict(X_test)
    model_results.append({{"name": name,
                           "mae": float(mean_absolute_error(y_test, pred)),
                           "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                           "r2": float(r2_score(y_test, pred))}})
model_results.sort(key=lambda x: x["mae"])
best_model_name = model_results[0]["name"]
best_model = models[best_model_name]
result = {{"status":"ok","best_model":best_model_name,
           "best_mae":model_results[0]["mae"],"best_rmse":model_results[0]["rmse"],
           "best_r2":model_results[0]["r2"],"all_models":model_results,
           "train_size":len(X_train),"test_size":len(X_test)}}

--- АНТИПРИМЕР (не делай так) ---
# model_results = [...]  # ОШИБКА: хардкодить результаты без реального обучения
# best_model = RandomForestRegressor()  # ОШИБКА: не обученная модель
# Не используй feature_importances_ без проверки hasattr

ТРЕБОВАНИЯ:
- Используй только переменные df, feature_cols (уже доступны), импортируй sklearn
- Сохрани best_model, best_model_name, model_results, result
- Не используй markdown-форматирование
- Верни ТОЛЬКО Python-код, без пояснений
"""

    response = llm.invoke(prompt)
    logger.info("LLM сгенерировал код обучения моделей")

    try:
        local_vars = {"df": df, "feature_cols": feature_cols, "np": np}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        best_model = local_vars.get("best_model")
        best_model_name = local_vars.get("best_model_name")
        model_results = local_vars.get("model_results", [])

        if best_model is None:
            raise ValueError("LLM-код не создал переменную best_model")

        STATE["best_model"] = best_model
        STATE["best_model_name"] = best_model_name
        STATE["model_results"] = model_results

        result = local_vars.get("result", {})

        # Долгосрочная память: сохраняем модель если лучше предыдущей
        os.makedirs(MEMORY_DIR, exist_ok=True)
        model_path = os.path.join(MEMORY_DIR, "best_model.pkl")
        metrics_path = os.path.join(MEMORY_DIR, "best_metrics.json")

        prev_mae = float("inf")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    prev = json.load(f)
                    prev_mae = prev.get("best_mae", float("inf"))
            except Exception:
                pass

        current_mae = result.get("best_mae", float("inf"))
        if current_mae < prev_mae:
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            result["model_saved"] = True
            result["previous_mae"] = prev_mae
            result["improvement"] = prev_mae - current_mae
            logger.info(f"Модель сохранена | MAE {prev_mae:,.0f} → {current_mae:,.0f}")
        else:
            result["model_saved"] = False
            result["previous_mae"] = prev_mae
            logger.info(f"Модель не обновлена | текущий MAE {current_mae:,.0f} >= предыдущего {prev_mae:,.0f}")

        log_action("train_and_compare_models", f"Лучшая: {best_model_name}, MAE={current_mae:,.0f}")
        logger.info(f"Train завершён | best={best_model_name} | MAE={current_mae:,.0f}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Train ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
