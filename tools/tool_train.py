"""
tools/tool_train.py
Tool 3: Обучение трёх моделей, подбор гиперпараметров, сравнение, выбор лучшей.
"""

import json
import traceback
import os
import pickle
import numpy as np

from langchain_core.tools import tool
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.train")

MEMORY_DIR = "memory"


@tool
def train_and_compare_models(dummy: str = "") -> str:
    """
    Обучает три ML модели (Ridge, RandomForest, GradientBoosting),
    выполняет подбор гиперпараметров GridSearchCV для лучшей,
    сравнивает по MAE/RMSE/R² и сохраняет лучшую.
    Вызывай после preprocess_data.
    """
    logger.info("Начало обучения моделей")
    if STATE["df_processed"] is None:
        logger.error("df_processed не найден в STATE")
        return json.dumps({"status": "error", "message": "Сначала вызови preprocess_data"})

    df = STATE["df_processed"]
    feature_cols = STATE["feature_cols"]

    try:
        X = df[feature_cols].values
        y = df["salary_target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        }

        model_results = []
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            model_results.append({"name": name, "mae": mae, "rmse": rmse, "r2": r2})
            trained_models[name] = model
            logger.info(f"  {name} | MAE={mae:,.0f} | RMSE={rmse:,.0f} | R2={r2:.4f}")

        model_results.sort(key=lambda x: x["mae"])
        best_name = model_results[0]["name"]
        best_model = trained_models[best_name]

        # Подбор гиперпараметров для лучшей модели
        logger.info(f"GridSearchCV для {best_name}...")
        param_grids = {
            "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]},
            "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 4, 5]},
        }

        grid = GridSearchCV(
            best_model, param_grids[best_name],
            cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        tuned_model = grid.best_estimator_
        y_pred_tuned = tuned_model.predict(X_test)
        tuned_mae = float(mean_absolute_error(y_test, y_pred_tuned))
        tuned_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_tuned)))
        tuned_r2 = float(r2_score(y_test, y_pred_tuned))

        logger.info(f"  {best_name} (tuned) | MAE={tuned_mae:,.0f} | params={grid.best_params_}")

        if tuned_mae < model_results[0]["mae"]:
            best_model = tuned_model
            model_results[0].update({"mae": tuned_mae, "rmse": tuned_rmse, "r2": tuned_r2, "tuned": True})
        model_results[0]["best_params"] = grid.best_params_

        STATE["best_model"] = best_model
        STATE["best_model_name"] = best_name
        STATE["model_results"] = model_results

        result = {
            "status": "ok",
            "best_model": best_name,
            "best_mae": model_results[0]["mae"],
            "best_rmse": model_results[0]["rmse"],
            "best_r2": model_results[0]["r2"],
            "best_params": grid.best_params_,
            "all_models": model_results,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        os.makedirs(MEMORY_DIR, exist_ok=True)
        model_path = os.path.join(MEMORY_DIR, "best_model.pkl")
        prev_mae = float("inf")
        metrics_path = os.path.join(MEMORY_DIR, "best_metrics.json")
        if os.path.exists(metrics_path):
            try:
                prev_mae = json.load(open(metrics_path)).get("best_mae", float("inf"))
            except Exception:
                pass

        current_mae = result["best_mae"]
        if current_mae < prev_mae:
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            result["model_saved"] = True
            result["improvement"] = prev_mae - current_mae
        else:
            result["model_saved"] = False
        result["previous_mae"] = prev_mae

        log_action("train_and_compare_models", f"Лучшая: {best_name}, MAE={current_mae:,.0f}")
        logger.info(f"Обучение завершено | best={best_name} | MAE={current_mae:,.0f} | R2={result['best_r2']:.4f} | saved={result['model_saved']}")
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error(f"Обучение ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
