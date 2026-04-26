"""
tools/state.py
Общее состояние между шагами пайплайна.
Все tools читают и пишут сюда.
"""
 
STATE = {
    "df_processed": None,   # DataFrame после предобработки
    "feature_cols": None,   # список признаков
    "best_model": None,     # обученная лучшая модель
    "best_model_name": None,
    "model_results": None,  # результаты всех трёх моделей
}