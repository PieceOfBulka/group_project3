"""
agent.py
ReAct агент для предсказания зарплат HH.ru.
"""

import os
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import ALL_TOOLS

load_dotenv()

model = ChatOpenAI(
    model=os.getenv('MODEL_NAME', 'minimax/minimax-m2.5:free'),
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('API_KEY'),
)

agent = create_agent(
    model=model,
    tools=ALL_TOOLS,
    system_prompt="""Ты — ML агент для предсказания зарплат на основе данных HH.ru.

Используй инструменты СТРОГО по порядку:
1. load_and_explore_data(filepath) — изучи данные
2. preprocess_data(filepath) — предобработай данные
3. train_and_compare_models(dummy="") — обучи и сравни модели
4. predict_salary(vacancy_json) — предскажи зарплату

После каждого шага кратко объясни результат по-русски.
В конце: какая модель лучшая и какая предсказанная зарплата.
""",
)


def run(csv_filepath: str, target_vacancy: dict) -> str:
    user_message = f"""Выполни пайплайн предсказания зарплат по порядку:
1. Загрузи данные: {csv_filepath}
2. Предобработай данные: {csv_filepath}
3. Обучи модели (dummy="")
4. Предскажи зарплату: {json.dumps(target_vacancy, ensure_ascii=False)}"""

    print("🚀 Агент запущен...\n")
    
    for step in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        for key, value in step.items():
            if key == "agent":
                msg = value["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"🔧 Вызываю tool: {tc['name']}")
                elif msg.content:
                    print(f"🤖 Агент: {msg.content}")
            elif key == "tools":
                msg = value["messages"][-1]
                print(f"✅ Результат tool:\n{msg.content[:300]}...\n")
    
    answer = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return answer["messages"][-1].content


if __name__ == "__main__":
    TARGET_VACANCY = {
        "name": "Middle Python Developer",
        "experience": "between1And3",
        "employment": "full",
        "schedule": "remote",
        "city": "Москва",
        "skills": "Python;FastAPI;PostgreSQL;Docker;Git",
    }

    result = run(csv_filepath="data/sample_data.csv", target_vacancy=TARGET_VACANCY)
    print(result)