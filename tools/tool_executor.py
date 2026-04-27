"""
tools/tool_executor.py
Tool 0: Универсальный исполнитель — принимает skill-промпт, генерирует Python-код через LLM и выполняет его.
Агент вызывает этот инструмент напрямую, передавая подробный промпт с описанием задачи.
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

logger = get_logger("tool.executor")


@tool
def execute_skill(skill_prompt: str) -> str:
    """
    Универсальный исполнитель навыков (skills). Принимает подробный текстовый промпт-инструкцию,
    генерирует Python-код с помощью LLM и выполняет его с авто-исправлением ошибок (до 3 попыток).

    Используй этот инструмент когда нужно выполнить произвольную аналитическую задачу:
    - дополнительный анализ данных
    - кастомная визуализация или статистика
    - проверка качества данных
    - любые вычисления поверх уже загруженных данных

    В промпте укажи:
    - какие данные доступны (df_raw, df_processed, feature_cols, best_model из STATE)
    - что именно нужно вычислить или сделать
    - какой результат ожидается (переменная result — dict)

    Аргумент skill_prompt: строка с полным описанием задачи на русском или английском.
    """
    logger.info(f"execute_skill | prompt_len={len(skill_prompt)}")
    llm = get_llm()

    # Собираем контекст из STATE для подстановки в промпт
    state_context = {}
    if STATE.get("df_raw") is not None:
        df_raw = STATE["df_raw"]
        state_context["df_raw_shape"] = df_raw.shape
        state_context["df_raw_columns"] = list(df_raw.columns)

    if STATE.get("df_processed") is not None:
        df_proc = STATE["df_processed"]
        state_context["df_processed_shape"] = df_proc.shape
        state_context["feature_cols"] = STATE.get("feature_cols", [])

    if STATE.get("best_model_name") is not None:
        state_context["best_model_name"] = STATE["best_model_name"]
        state_context["model_results"] = STATE.get("model_results", [])

    full_prompt = f"""Ты — опытный Data Scientist. Выполни следующую задачу, написав Python-код.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже готовы в окружении, не переопределяй):
- pd — pandas
- np — numpy
- STATE — словарь с данными агента: df_raw, df_processed, feature_cols, best_model, model_results

ТЕКУЩЕЕ СОСТОЯНИЕ STATE:
{json.dumps(state_context, ensure_ascii=False, indent=2, default=str)}

ЗАДАЧА:
{skill_prompt}

ТРЕБОВАНИЯ:
- Сохрани итог в переменную result (dict) с ключом "status": "ok" и нужными данными
- Если нужен df_raw — получи его как: df = STATE["df_raw"]
- Если нужен df_processed — получи его как: df = STATE["df_processed"]
- Если нужна модель — получи её как: model = STATE["best_model"]
- Не используй markdown-форматирование в коде
- Верни ТОЛЬКО Python-код, без пояснений
"""

    response = llm.invoke(full_prompt)
    logger.info("LLM сгенерировал код для execute_skill")

    try:
        local_vars = {"pd": pd, "np": np, "STATE": STATE, "json": json}
        exec_llm_code_with_retry(response.content, local_vars, llm)

        result = local_vars.get("result", {"status": "ok", "message": "Код выполнен успешно"})
        log_action("execute_skill", f"Выполнен skill: {skill_prompt[:80]}...")
        logger.info(f"execute_skill завершён | status={result.get('status')}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"execute_skill ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
