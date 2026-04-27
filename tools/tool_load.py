"""
tools/tool_load.py
Tool 1: Загрузка и EDA — агент сам пишет код через LLM.
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

logger = get_logger("tool.load")


@tool
def load_and_explore_data(filepath: str) -> str:
    """
    Загружает CSV-файл с вакансиями HH.ru и выполняет первичный анализ данных (EDA).
    Агент самостоятельно пишет Python-код для анализа.
    Аргумент: filepath — путь к CSV-файлу.
    """
    logger.info(f"EDA | filepath={filepath}")
    llm = get_llm()

    # Читаем мета-информацию для контекста промпта
    try:
        _df_meta = pd.read_csv(filepath, nrows=3)
        columns_info = list(_df_meta.columns)
        dtypes_info = _df_meta.dtypes.astype(str).to_dict()
        sample_info = _df_meta.head(2).to_dict("records")
    except Exception:
        columns_info, dtypes_info, sample_info = [], {}, []

    prompt = f"""Ты — Data Engineer. Напиши Python-код для загрузки и первичного анализа (EDA) CSV-файла с вакансиями HH.ru.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже импортированы):
- pd — pandas
- np — numpy
- filepath — строка с путём к файлу: "{filepath}"

МЕТА-ИНФОРМАЦИЯ О ФАЙЛЕ:
- Известные колонки: {columns_info}
- Типы данных: {dtypes_info}
- Первые строки (sample): {sample_info}

ЗАДАЧА — написать код, который:
1. Загружает CSV: df = pd.read_csv(filepath)
2. Вычисляет базовую статистику
3. Находит колонки с зарплатой (содержат 'salary' в названии)
4. Считает пропуски по каждой колонке
5. Анализирует распределение зарплат (min, max, mean, median)
6. Формирует словарь result со следующими ключами:
   - "status": "ok"
   - "rows": int — количество строк
   - "cols": int — количество колонок
   - "columns": list — список всех колонок
   - "salary_columns": list — колонки с зарплатой
   - "missing_values": dict — {{колонка: кол-во пропусков}} только для колонок с пропусками
   - "salary_stats": dict — {{"min": float, "max": float, "mean": float, "median": float}}
     (вычисли из salary_from и salary_to если есть, иначе из salary)
   - "dtypes": dict — {{колонка: тип}}
   - "sample": list — первые 3 строки как df.head(3).to_dict('records')
   - "summary": str — 1-2 предложения описания датасета

--- FEW-SHOT ПРИМЕР (правильный код) ---
import pandas as pd
import numpy as np
df = pd.read_csv(filepath)
salary_columns = [col for col in df.columns if 'salary' in col.lower()]
missing = {{col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}}
sal = pd.concat([df[c] for c in salary_columns if c in df.columns], ignore_index=True).dropna()
result = {{
    "status": "ok",
    "rows": len(df),
    "cols": len(df.columns),
    "columns": list(df.columns),
    "salary_columns": salary_columns,
    "missing_values": missing,
    "salary_stats": {{"min": float(sal.min()), "max": float(sal.max()), "mean": float(sal.mean()), "median": float(sal.median())}},
    "dtypes": df.dtypes.astype(str).to_dict(),
    "sample": df.head(3).to_dict('records'),
    "summary": f"Датасет {{len(df)}} строк x {{len(df.columns)}} колонок. Зарплатные колонки: {{salary_columns}}."
}}

--- АНТИПРИМЕР (не делай так) ---
# result["rows"] = ...  # ОШИБКА: result не создан как dict заранее
# import pandas as [pd](url)  # ОШИБКА: markdown в коде
# df.missing_count()  # ОШИБКА: несуществующий метод

ТРЕБОВАНИЯ:
- Используй только переменные, которые сам объявил или которые доступны (pd, np, filepath)
- Не используй markdown-форматирование внутри кода
- Верни ТОЛЬКО Python-код, без пояснений и без ```python блоков
"""

    response = llm.invoke(prompt)
    logger.info("LLM сгенерировал код EDA")

    try:
        local_vars = {"pd": pd, "np": np, "filepath": filepath}
        exec_llm_code_with_retry(response.content, local_vars, llm)
        result = local_vars.get("result", {})
        df_loaded = local_vars.get("df")
        if df_loaded is not None:
            STATE["df_raw"] = df_loaded
        log_action("load_and_explore_data", f"Загружено {result.get('rows', '?')} строк, {result.get('cols', '?')} колонок")
        logger.info(f"EDA завершён | rows={result.get('rows')} | cols={result.get('cols')}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"EDA ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
