"""
tools/tool_report.py
Tool 5: Генерация HTML-отчёта и сохранение метрик.
"""

import json
import os
from datetime import datetime
from langchain_core.tools import tool

from tools.state import STATE


REPORTS_DIR = "reports"
MEMORY_DIR = "memory"


@tool
def generate_report(dummy: str = "") -> str:
    """
    Генерирует HTML-отчёт с результатами работы агента и сохраняет метрики в JSON.
    Вызывай после train_and_compare_models.
    """
    if STATE["model_results"] is None:
        return json.dumps({"status": "error", "message": "Сначала вызови train_and_compare_models"})

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)

    model_results = STATE["model_results"]
    best_model_name = STATE["best_model_name"]
    action_history = STATE.get("action_history", [])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    prev_metrics = None
    prev_metrics_path = os.path.join(MEMORY_DIR, "best_metrics.json")
    if os.path.exists(prev_metrics_path):
        with open(prev_metrics_path) as f:
            prev_metrics = json.load(f)

    best = next(m for m in model_results if m["name"] == best_model_name)
    improved = None
    if prev_metrics:
        prev_mae = prev_metrics.get("best_mae", float("inf"))
        improved = best["mae"] < prev_mae
        comparison_text = (
            f"<p>Предыдущий лучший MAE: <b>{prev_mae:,.0f} ₽</b> ({prev_metrics.get('best_model', '—')})<br>"
            f"Текущий лучший MAE: <b>{best['mae']:,.0f} ₽</b> ({best_model_name})<br>"
            f"<b style='color:{'green' if improved else 'red'}'>{'✅ Улучшение!' if improved else '⚠️ Регрессия качества'}</b></p>"
        )
    else:
        comparison_text = "<p>Первый запуск — предыдущих результатов нет.</p>"

    rows_html = ""
    for m in model_results:
        badge = " 🏆" if m["name"] == best_model_name else ""
        rows_html += f"""
        <tr>
            <td><b>{m['name']}{badge}</b></td>
            <td>{m['mae']:,.0f} ₽</td>
            <td>{m['rmse']:,.0f} ₽</td>
            <td>{m['r2']:.4f}</td>
        </tr>"""

    history_html = ""
    for step in action_history:
        history_html += f"<li><b>{step['tool']}</b> — {step['time']}</li>"
    if not history_html:
        history_html = "<li>История действий недоступна</li>"

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Отчёт ML Агента — Предсказание зарплат HH.ru</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
  h1 {{ color: #1a56db; }}
  h2 {{ color: #1e429f; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #1a56db; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 10px; border-bottom: 1px solid #e5e7eb; }}
  tr:hover {{ background: #f9fafb; }}
  .badge {{ background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }}
  .meta {{ color: #6b7280; font-size: 0.9em; }}
  .highlight {{ background: #eff6ff; padding: 16px; border-radius: 8px; border-left: 4px solid #1a56db; }}
</style>
</head>
<body>
<h1>📊 Отчёт ML Агента: Предсказание зарплат HH.ru</h1>
<p class="meta">Сгенерирован: {timestamp} | Run ID: {run_id}</p>

<h2>🤖 Сравнение ML-моделей</h2>
<table>
  <thead><tr><th>Модель</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>

<div class="highlight">
  <b>Лучшая модель:</b> {best_model_name}<br>
  MAE = {best['mae']:,.0f} ₽ | RMSE = {best['rmse']:,.0f} ₽ | R² = {best['r2']:.4f}
</div>

<h2>📈 Сравнение с предыдущим запуском</h2>
{comparison_text}

<h2>🔍 История действий агента (кратковременная память)</h2>
<ol>{history_html}</ol>

<h2>💡 Бизнес-выводы</h2>
<ul>
  <li>Средняя ошибка предсказания зарплаты составляет ~{best['mae']:,.0f} ₽ — приемлемо для HR-аналитики.</li>
  <li>Модель {best_model_name} показала лучший результат среди трёх протестированных.</li>
  <li>R² = {best['r2']:.2f}: модель объясняет {best['r2']*100:.0f}% дисперсии зарплат.</li>
  <li>Ключевые факторы: опыт работы, город, стек технологий.</li>
</ul>
</body>
</html>"""

    report_path = os.path.join(REPORTS_DIR, f"report_{run_id}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    metrics = {
        "run_id": run_id,
        "timestamp": timestamp,
        "best_model": best_model_name,
        "best_mae": best["mae"],
        "best_rmse": best["rmse"],
        "best_r2": best["r2"],
        "all_models": model_results,
    }
    with open(prev_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    history_path = os.path.join(MEMORY_DIR, "run_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return json.dumps({
        "status": "ok",
        "report_path": report_path,
        "metrics_saved": prev_metrics_path,
        "improved": improved,
        "best_model": best_model_name,
        "best_mae": best["mae"],
    }, ensure_ascii=False)
