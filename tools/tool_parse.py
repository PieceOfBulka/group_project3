"""
tools/tool_parse.py
Tool 0: Парсинг вакансий с HH.ru (2 страницы) + обогащение через LLM.
Парсит свежие вакансии через публичный API HH.ru, извлекает навыки/уровень/категорию/зарплату
с помощью LLM и дописывает новые строки в основной CSV-датасет.
"""

import json
import os
import time
import traceback
import requests
import pandas as pd
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.parse")

HH_API_URL = "https://api.hh.ru/vacancies"
DATA_FILE = "data/hh_it_5000_final.csv"
HH_HEADERS = {"User-Agent": "hse-salary-agent/1.0 (educational project)"}

ENRICH_PROMPT = """Ты — система извлечения структурированных данных из вакансий.
Извлеки данные из вакансии и верни строго валидный JSON без markdown и комментариев.

Вакансия:
title: {title}
experience: {experience}
employment: {employment}
schedule: {schedule}
salary_raw: {salary_raw}
description: {description}

Верни JSON:
{{
  "skills": [],
  "level": "unknown",
  "category": "Other",
  "salary_from": null,
  "salary_to": null,
  "currency": "unknown"
}}

Правила:
- skills: только реальные технологии из текста (Python, SQL, Docker и т.д.), максимум 15, без soft skills
- level: junior / middle / senior / lead / head / unknown (по опыту и названию)
- category: AI Engineer / Data Scientist / ML Engineer / Data Engineer / Backend Developer /
  Frontend Developer / Fullstack Developer / DevOps Engineer / QA Engineer / Business Analyst /
  Product Manager / Other
- salary_from / salary_to: числа или null (RUR, USD, EUR → currency)
- currency: RUR / USD / EUR / unknown
Ответ — только JSON, без пояснений."""


def _fetch_page(query: str, page: int) -> list[dict]:
    """Запрашивает одну страницу вакансий через API HH.ru."""
    params = {
        "text": query,
        "area": 113,           # Россия
        "per_page": 20,
        "page": page,
        "only_with_salary": False,
        "search_field": "name",
        "order_by": "publication_time",
    }
    try:
        r = requests.get(HH_API_URL, params=params, headers=HH_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json().get("items", [])
    except Exception as e:
        logger.warning(f"Ошибка запроса страницы {page}: {e}")
        return []


def _fetch_description(vacancy_id: str) -> str:
    """Загружает описание конкретной вакансии."""
    try:
        r = requests.get(f"{HH_API_URL}/{vacancy_id}", headers=HH_HEADERS, timeout=8)
        r.raise_for_status()
        data = r.json()
        # Убираем HTML-теги простой заменой
        desc = data.get("description", "") or ""
        import re
        desc = re.sub(r"<[^>]+>", " ", desc)
        desc = re.sub(r"\s+", " ", desc).strip()
        return desc[:3000]
    except Exception:
        return ""


def _enrich_vacancy(row: dict, llm) -> dict:
    """Обогащает одну вакансию через LLM — извлекает skills, level, category, salary."""
    salary = row.get("salary") or {}
    salary_raw = ""
    if salary:
        frm = salary.get("from")
        to = salary.get("to")
        cur = salary.get("currency", "")
        if frm and to:
            salary_raw = f"{frm}–{to} {cur}"
        elif frm:
            salary_raw = f"от {frm} {cur}"
        elif to:
            salary_raw = f"до {to} {cur}"

    prompt = ENRICH_PROMPT.format(
        title=row.get("name", ""),
        experience=(row.get("experience") or {}).get("name", ""),
        employment=(row.get("employment") or {}).get("name", ""),
        schedule=(row.get("schedule") or {}).get("name", ""),
        salary_raw=salary_raw or "не указана",
        description=row.get("_description", ""),
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Вырезаем JSON если модель всё-таки добавила markdown
        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        enriched = json.loads(content)
    except Exception as e:
        logger.warning(f"Ошибка обогащения '{row.get('name')}': {e}")
        enriched = {"skills": [], "level": "unknown", "category": "Other",
                    "salary_from": None, "salary_to": None, "currency": "unknown"}

    return enriched


def _vacancy_to_row(item: dict, enriched: dict) -> dict:
    """Преобразует API-ответ + enriched данные в строку датафрейма."""
    salary = item.get("salary") or {}
    area = (item.get("area") or {}).get("name", "")
    employer = (item.get("employer") or {}).get("name", "")
    experience_id = (item.get("experience") or {}).get("id", "")
    employment_id = (item.get("employment") or {}).get("id", "")
    schedule_id = (item.get("schedule") or {}).get("id", "")

    skills_list = enriched.get("skills", [])
    skills_str = ";".join(skills_list) if isinstance(skills_list, list) else ""

    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "salary_from": enriched.get("salary_from") or salary.get("from"),
        "salary_to": enriched.get("salary_to") or salary.get("to"),
        "salary_currency": enriched.get("currency") or salary.get("currency", ""),
        "experience": experience_id,
        "employment": employment_id,
        "schedule": schedule_id,
        "city": area,
        "employer": employer,
        "skills": skills_str,
        "level": enriched.get("level", "unknown"),
        "category": enriched.get("category", "Other"),
        "published_at": item.get("published_at", ""),
        "url": item.get("alternate_url", ""),
    }


@tool
def parse_hh_vacancies(query: str = "Data Scientist Python ML Engineer") -> str:
    """
    Парсит свежие вакансии с HH.ru (2 страницы ~40 вакансий) по поисковому запросу,
    обогащает их через LLM (извлекает навыки, уровень, категорию, зарплату)
    и дописывает новые строки в основной CSV-датасет.
    Аргумент query: поисковый запрос (например, 'Data Scientist Python').
    """
    logger.info(f"Парсинг HH.ru | query='{query}'")
    llm = get_llm()

    # 1. Загружаем существующий датасет
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        existing_ids = set(df_existing["id"].astype(str).tolist()) if "id" in df_existing.columns else set()
    else:
        df_existing = pd.DataFrame()
        existing_ids = set()
    logger.info(f"Существующий датасет: {len(df_existing)} строк, {len(existing_ids)} уникальных ID")

    # 2. Парсим 2 страницы
    all_items = []
    for page in range(2):
        items = _fetch_page(query, page)
        all_items.extend(items)
        logger.info(f"Страница {page}: получено {len(items)} вакансий")
        time.sleep(0.3)

    # 3. Фильтруем уже существующие
    new_items = [v for v in all_items if str(v.get("id", "")) not in existing_ids]
    logger.info(f"Новых вакансий для добавления: {len(new_items)} из {len(all_items)}")

    if not new_items:
        log_action("parse_hh_vacancies", "Новых вакансий не найдено")
        return json.dumps({
            "status": "ok",
            "message": "Новых вакансий не найдено — датасет актуален",
            "filepath": DATA_FILE,
            "rows_existing": len(df_existing),
            "new_added": 0,
        }, ensure_ascii=False)

    # 4. Загружаем описания и обогащаем через LLM
    new_rows = []
    for i, item in enumerate(new_items):
        vac_id = str(item.get("id", ""))
        logger.info(f"Обогащение {i+1}/{len(new_items)}: {item.get('name', '')}")

        desc = _fetch_description(vac_id)
        item["_description"] = desc
        time.sleep(0.2)

        enriched = _enrich_vacancy(item, llm)
        row = _vacancy_to_row(item, enriched)
        new_rows.append(row)
        time.sleep(0.1)

    # 5. Дописываем в датасет
    df_new = pd.DataFrame(new_rows)

    if not df_existing.empty:
        # Выравниваем колонки
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[[c for c in df_existing.columns if c in df_new.columns]]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df_combined.to_csv(DATA_FILE, index=False, encoding="utf-8")

    # Обновляем STATE
    STATE["df_raw"] = df_combined

    log_action("parse_hh_vacancies", f"Добавлено {len(new_rows)} вакансий, итого {len(df_combined)}")
    logger.info(f"Парсинг завершён | новых={len(new_rows)} | итого={len(df_combined)}")

    return json.dumps({
        "status": "ok",
        "filepath": DATA_FILE,
        "rows_existing": len(df_existing),
        "new_added": len(new_rows),
        "rows_total": len(df_combined),
        "query": query,
        "sample_new": [r["name"] for r in new_rows[:5]],
        "message": f"Добавлено {len(new_rows)} новых вакансий. Итого в датасете: {len(df_combined)}",
    }, ensure_ascii=False)
