"""
tools/tool_report.py
Tool 5: Генерация HTML-отчёта — агент сам пишет код через LLM.
"""

import json
import os
import traceback
from datetime import datetime
from langchain_core.tools import tool

from tools.llm import get_llm
from tools.executor import exec_llm_code_with_retry
from tools.state import STATE, log_action
from tools.logger import get_logger

logger = get_logger("tool.report")

REPORTS_DIR = "reports"
MEMORY_DIR = "memory"


@tool
def generate_report(dummy: str = "") -> str:
    """
    Генерирует HTML-отчёт с результатами работы агента и сохраняет метрики в JSON.
    Вызывай после train_and_compare_models.
    Аргумент dummy можно передать пустым "".
    """
    if STATE.get("model_results") is None:
        return json.dumps({"status": "error", "message": "Сначала вызови train_and_compare_models"})

    model_results = STATE["model_results"]
    best_model_name = STATE["best_model_name"]
    action_history = STATE.get("action_history", [])

    # Загружаем предыдущие метрики для сравнения
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)

    metrics_path = os.path.join(MEMORY_DIR, "best_metrics.json")
    prev_metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                prev_metrics = json.load(f)
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    llm = get_llm()

    prompt = f"""Ты — веб-разработчик и аналитик данных. Напиши Python-код для генерации HTML-отчёта о результатах ML-агента.

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ (уже готовы):
- model_results — список dict: {json.dumps(model_results, ensure_ascii=False)}
- best_model_name — имя лучшей модели: "{best_model_name}"
- action_history — список шагов агента: {json.dumps(action_history, ensure_ascii=False)}
- prev_metrics — метрики предыдущего запуска (или None): {json.dumps(prev_metrics, ensure_ascii=False)}
- timestamp — строка времени: "{timestamp}"
- run_id — ID запуска: "{run_id}"
- REPORTS_DIR — папка для отчётов: "{REPORTS_DIR}"
- MEMORY_DIR — папка памяти: "{MEMORY_DIR}"
- os — модуль os
- json — модуль json

ЗАДАЧА — напиши код, который:

1. НАЙДИ ЛУЧШУЮ МОДЕЛЬ:
   - best = следующий элемент из model_results где name == best_model_name
   - Если не найден — возьми первый элемент

2. СРАВНЕНИЕ С ПРЕДЫДУЩИМ ЗАПУСКОМ:
   - Если prev_metrics не None: сравни best["mae"] с prev_metrics["best_mae"]
   - improved = True если текущий MAE меньше предыдущего
   - Сформируй comparison_html — параграф с информацией о сравнении

3. ТАБЛИЦА МОДЕЛЕЙ:
   - Пройдись по model_results
   - Для лучшей модели добавь маркер "★"
   - Для каждой строки: имя, MAE (в рублях с разделителем тысяч), RMSE, R²

4. ИСТОРИЯ ДЕЙСТВИЙ:
   - Из action_history создай нумерованный список <ol>
   - Каждый элемент: tool + time из dict шага

5. БИЗНЕС-ВЫВОДЫ:
   - 4 bullet-пункта на русском, основанные на реальных значениях метрик

6. СГЕНЕРИРУЙ HTML:
   - Полноценный HTML5 документ с UTF-8
   - Стильный CSS (синяя палитра #1a56db, таблица с hover, карточки)
   - Разделы: заголовок, таблица моделей, highlight-блок с лучшей моделью,
     сравнение с предыдущим, история действий, бизнес-выводы
   - Сохрани в переменную html_content — строку

7. СОХРАНИ ФАЙЛЫ:
   - report_path = os.path.join(REPORTS_DIR, f"report_{{run_id}}.html")
   - Запиши html_content в report_path (encoding="utf-8")
   - Обнови best_metrics.json: run_id, timestamp, best_model, best_mae, best_rmse, best_r2, all_models
   - Дозапиши в run_history.jsonl (режим "a")

8. РЕЗУЛЬТАТ:
   - result = {{
       "status": "ok",
       "report_path": report_path,
       "best_model": best_model_name,
       "best_mae": float(best["mae"]),
       "improved": improved (bool или None если нет предыдущих данных),
     }}

--- FEW-SHOT ПРИМЕР (структура HTML) ---
html_content = f\"\"\"<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>ML Agent Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
  h1 {{ color: #1a56db; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #1a56db; color: white; padding: 10px; }}
  td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f9fafb; }}
  .highlight {{ background: #eff6ff; padding: 16px; border-radius: 8px; border-left: 4px solid #1a56db; }}
</style>
</head>
<body>
<h1>Отчёт ML Агента: Предсказание зарплат HH.ru</h1>
...
</body>
</html>\"\"\"

--- АНТИПРИМЕР ---
# html_content = "<html></html>"  # ОШИБКА: пустой шаблон без данных
# open(report_path, "w")  # ОШИБКА: без encoding="utf-8"

ТРЕБОВАНИЯ:
- Используй только переменные из списка выше (model_results, best_model_name, и т.д.)
- HTML должен содержать реальные числа из model_results, а не placeholder'ы
- Все числа MAE/RMSE форматируй как целые рубли с пробелами-разделителями
- Верни ТОЛЬКО Python-код, без пояснений и без markdown
"""

    response = llm.invoke(prompt)
    logger.info("LLM сгенерировал код генерации отчёта")

    try:
        local_vars = {
            "model_results": model_results,
            "best_model_name": best_model_name,
            "action_history": action_history,
            "prev_metrics": prev_metrics,
            "timestamp": timestamp,
            "run_id": run_id,
            "REPORTS_DIR": REPORTS_DIR,
            "MEMORY_DIR": MEMORY_DIR,
            "os": os,
            "json": json,
        }
        exec_llm_code_with_retry(response.content, local_vars, llm)

        result = local_vars.get("result", {})
        if not result:
            result = {
                "status": "ok",
                "report_path": os.path.join(REPORTS_DIR, f"report_{run_id}.html"),
                "best_model": best_model_name,
            }

        log_action("generate_report", f"Отчёт сохранён: {result.get('report_path','?')}")
        logger.info(f"Report завершён | path={result.get('report_path')}")
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Report ошибка | {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e), "traceback": traceback.format_exc()})
