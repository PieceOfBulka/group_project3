"""
tools/tool_load.py
Tool 1: Загрузка и первичный анализ CSV с HH.ru.
"""

import json
import traceback
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import log_action


@tool
def load_and_explore_data(filepath: str) -> str:
    """
    Загружает CSV файл с вакансиями HH.ru и возвращает подробную статистику.
    Используй этот tool первым. Аргумент: filepath — путь к CSV файлу.
    """
    llm = get_llm()

    prompt = f"""Ты — Data Engineer. Напиши Python-код для загрузки и анализа CSV файла с вакансиями HH.ru.

ЗАДАЧА:
1. Загрузи CSV файл по пути: {filepath}
2. Сохрани результат в переменную `result` как словарь со следующими ключами:
   - "rows": количество строк (int)
   - "columns": список названий колонок (list)
   - "salary_columns": список колонок связанных с зарплатой (list) — ищи колонки где в названии есть "salary" или "зарплата"
   - "missing_values": словарь {{колонка: кол-во пропусков}} только для колонок с пропусками
   - "sample": первые 3 строки — df.head(3).to_dict('records')
   - "summary": краткое описание датасета (строка)

ВАЖНО:
- import pandas as pd
- НЕ используй переменные которые не объявил сам
- Определяй salary_columns через простой перебор: [col for col in df.columns if 'salary' in col.lower()]
- Возвращай только Python-код, без пояснений, без markdown-блоков
"""

    response = llm.invoke(prompt)

    try:
        local_vars = {}
        exec_llm_code_with_retry(response.content, local_vars, llm)
        result = local_vars.get("result", {})
        log_action("load_and_explore_data", f"Загружено {result.get('rows', '?')} строк, {len(result.get('columns', []))} колонок")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})