"""
tools/executor.py
Утилита для выполнения кода сгенерированного LLM.
"""
import pandas as pd
import numpy as np
from tools.logger import get_logger

logger = get_logger("executor")

def clean_code(code: str) -> str:
    import re
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)
    
    # Убираем markdown-ссылки вида [text](url) → text
    code = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', code)
    
    return code


def exec_llm_code(code: str, local_vars: dict) -> dict:
    code = clean_code(code)
    exec(compile(code, "<llm_generated>", "exec"), local_vars)
    return local_vars


def exec_llm_code_with_retry(code: str, local_vars: dict, llm, max_retries: int = 3) -> dict:
    import traceback

    code = clean_code(code)
    original_vars = dict(local_vars)

    for attempt in range(max_retries):
        try:
            print(f"\n{'='*50}")
            print(f"📝 КОД (попытка {attempt + 1}):")
            print(code)
            print('='*50)
            
            local_vars_copy = dict(original_vars)

            exec(compile(code, "<llm_generated>", "exec"), local_vars_copy)
            local_vars.update(local_vars_copy)
            if attempt > 0:
                print(f"✅ Исправлено за {attempt + 1} попытки")
            return local_vars
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.warning(f"Попытка {attempt+1}/{max_retries} | ошибка: {e}")
            print(f"⚠️  Попытка {attempt + 1}/{max_retries} — ошибка: {e}")

            if attempt == max_retries - 1:
                logger.error(f"Код не исправлен за {max_retries} попытки | {e}")
                raise

            fix_prompt = f"""Этот Python-код содержит ошибку. Исправь её и верни только исправленный код.

Доступны готовые переменные: pd (pandas), np (numpy), и все переменные из local_vars.

КОД:
{code}

ОШИБКА:
{error_msg}

Верни только исправленный Python-код без пояснений и без markdown-блоков.
"""
            response = llm.invoke(fix_prompt)
            code = clean_code(response.content)
            logger.info(f"LLM исправила код, попытка {attempt+2}")
            print(f"🔄 LLM исправила код, повторяю...")

    return local_vars