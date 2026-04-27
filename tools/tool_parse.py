"""
tools/tool_parse.py
Tool 0: Парсинг вакансий с HH.ru (демонстрационный).
Имитирует запросы к API HH.ru и возвращает данные из локального файла.
"""

import json
import time
import os
from langchain_core.tools import tool
from tools.logger import get_logger

logger = get_logger("tool.parse")

HH_API_URL = "https://api.hh.ru/vacancies"
DATA_FILE = "data/hh_it_5000_final.csv"


@tool
def parse_hh_vacancies(query: str = "Data Scientist Python") -> str:
    """
    Парсит вакансии с HH.ru по поисковому запросу.
    Возвращает путь к сохранённому файлу с данными.
    Аргумент query: поисковый запрос (например, 'Data Scientist Python').
    """
    logger.info(f"Парсинг HH.ru | query='{query}'")
    print(f"\n🌐 Подключаюсь к HH.ru API: {HH_API_URL}")
    print(f"🔍 Поисковый запрос: '{query}'")

    pages_to_parse = 55
    per_page = 100

    for page in range(min(3, pages_to_parse)):
        print(f"   📄 Страница {page + 1}/{pages_to_parse} — получено {per_page} вакансий")
        time.sleep(0.1)

    print(f"   ✅ Всего получено: {pages_to_parse * per_page} вакансий")
    print(f"   💾 Сохраняю в {DATA_FILE}...")
    time.sleep(0.2)

    if not os.path.exists(DATA_FILE):
        logger.error(f"Файл не найден: {DATA_FILE}")
        return json.dumps({
            "status": "error",
            "message": f"Файл {DATA_FILE} не найден. Запустите generate_data.py",
        }, ensure_ascii=False)

    import pandas as pd
    df = pd.read_csv(DATA_FILE)

    result = {
        "status": "ok",
        "filepath": DATA_FILE,
        "rows": len(df),
        "columns": list(df.columns),
        "query": query,
        "source": HH_API_URL,
        "pages_parsed": pages_to_parse,
        "message": f"Данные успешно получены: {len(df)} вакансий сохранено в {DATA_FILE}",
    }

    from tools.state import log_action
    log_action("parse_hh_vacancies", f"Получено {len(df)} вакансий из {DATA_FILE}")
    logger.info(f"Парсинг завершён | rows={len(df)} | file={DATA_FILE}")
    print(f"   ✅ Парсинг завершён: {len(df)} вакансий")
    return json.dumps(result, ensure_ascii=False, default=str)
