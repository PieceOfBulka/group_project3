import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

model = ChatOpenAI(
    model=os.getenv('MODEL_NAME', 'minimax/minimax-m2.5:free'),
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('API_KEY')
)

agent = create_react_agent(
    model=model,
    tools=[],
    prompt='Ты умный помощник по написанию кода для ML проектов'
)

if __name__ == '__main__':
    answer = agent.invoke({'messages': [{'role':'user', 'content':'Напиши построение линейной регрессии'}]})
    print(answer['messages'][-1].content)