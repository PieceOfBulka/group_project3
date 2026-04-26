"""
tools/__init__.py
Собирает все tools в один список ALL_TOOLS.
В agent.py нужен только один импорт:

    from tools import ALL_TOOLS
"""

from tools.tool_load import load_and_explore_data
from tools.tool_preprocess import preprocess_data
from tools.tool_train import train_and_compare_models
from tools.tool_predict import predict_salary

ALL_TOOLS = [
    load_and_explore_data,
    preprocess_data,
    train_and_compare_models,
    predict_salary,
]