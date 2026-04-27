"""
tools/state.py
Общее состояние между шагами пайплайна (кратковременная память агента).
"""

from datetime import datetime

STATE = {
    "df_processed": None,
    "feature_cols": None,
    "best_model": None,
    "best_model_name": None,
    "model_results": None,
    "action_history": [],
}


def log_action(tool_name: str, summary: str = ""):
    STATE["action_history"].append({
        "tool": tool_name,
        "summary": summary,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
