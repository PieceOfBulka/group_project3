"""
tools/llm.py
Единое место для инициализации LLM.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv('MODEL_NAME', 'google/gemma-3-27b-it:free'),
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv('API_KEY'),
        temperature=0,
    )