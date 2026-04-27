"""
agent.py
ReAct-агент для предсказания зарплат HH.ru.
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """Ты — опытный ML-инженер, специализирующийся на анализе рынка труда.
Твоя задача — автономно выполнить полный пайплайн ML для предсказания зарплат по вакансиям HH.ru.

Используй инструменты СТРОГО по порядку:
1. parse_hh_vacancies(query) — собери данные с HH.ru
2. load_and_explore_data(filepath) — изучи загруженные данные
3. preprocess_data(filepath) — предобработай данные и выполни Feature Engineering
4. train_and_compare_models(dummy="") — обучи и сравни 3 ML-модели, сохрани лучшую
5. predict_salary(vacancy_json) — предскажи зарплату для целевой вакансии
6. generate_report(dummy="") — сформируй HTML-отчёт с метриками и выводами

После каждого шага кратко объясни результат по-русски.
Используй цепочку мыслей (Chain-of-Thought): перед каждым действием объясни, ПОЧЕМУ выбираешь именно этот инструмент.
В финальном ответе: какая модель лучшая, почему, и какая предсказанная зарплата.

Помни: ты работаешь как бизнес-аналитик — все выводы должны быть полезны HR-специалисту.
"""


def get_model(model_name: str | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or os.getenv("MODEL_NAME", "google/gemma-3-27b-it:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("API_KEY"),
        temperature=0.1,
    )


def run(
    csv_filepath: str = "data/hh_it_5000_final.csv",
    target_vacancy: dict | None = None,
    model_name: str | None = None,
) -> str:
    if target_vacancy is None:
        target_vacancy = {
            "name": "Middle Python Developer",
            "experience": "between1And3",
            "employment": "full",
            "schedule": "remote",
            "city": "Москва",
            "skills": "Python;FastAPI;PostgreSQL;Docker;Git",
        }

    llm = get_model(model_name)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)

    user_message = f"""Выполни полный ML-пайплайн для предсказания зарплат:

1. Спарси данные с HH.ru (query="Data Scientist Python ML Engineer")
2. Загрузи и изучи данные: {csv_filepath}
3. Предобработай данные: {csv_filepath}
4. Обучи и сравни модели
5. Предскажи зарплату для вакансии: {json.dumps(target_vacancy, ensure_ascii=False)}
6. Сформируй финальный отчёт

Целевая вакансия для предсказания:
{json.dumps(target_vacancy, ensure_ascii=False, indent=2)}"""

    print("🚀 Агент запущен...\n")

    final_content = ""
    for step in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        for key, value in step.items():
            if key == "agent":
                msg = value["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_preview = str(tc.get("args", ""))[:80]
                        print(f"🔧 Tool: {tc['name']}({args_preview})")
                elif msg.content:
                    print(f"\n🤖 Агент:\n{msg.content}\n")
                    final_content = msg.content
            elif key == "tools":
                msg = value["messages"][-1]
                preview = msg.content[:200].replace("\n", " ")
                print(f"✅ Результат: {preview}...\n")

    return final_content


if __name__ == "__main__":
    TARGET_VACANCY = {
        "name": "Middle Python Developer",
        "experience": "between1And3",
        "employment": "full",
        "schedule": "remote",
        "city": "Москва",
        "skills": "Python;FastAPI;PostgreSQL;Docker;Git",
    }

    result = run(csv_filepath="data/hh_it_5000_final.csv", target_vacancy=TARGET_VACANCY)
    print("\n" + "=" * 60)
    print("📋 ФИНАЛЬНЫЙ ОТВЕТ АГЕНТА:")
    print("=" * 60)
    print(result)
