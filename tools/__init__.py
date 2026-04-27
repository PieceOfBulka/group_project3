"""
tools/__init__.py
"""

from tools.tool_parse import parse_hh_vacancies
from tools.tool_load import load_and_explore_data
from tools.tool_preprocess import preprocess_data
from tools.tool_train import train_and_compare_models
from tools.tool_predict import predict_salary
from tools.tool_report import generate_report

ALL_TOOLS = [
    parse_hh_vacancies,
    load_and_explore_data,
    preprocess_data,
    train_and_compare_models,
    predict_salary,
    generate_report,
]
