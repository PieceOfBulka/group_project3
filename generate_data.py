"""
generate_data.py
Генерирует синтетический датасет вакансий HH.ru (5000+ строк).
Запускается один раз: python generate_data.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 5500

POSITIONS = [
    ("Python Developer", "Python;FastAPI;PostgreSQL;Docker;Git", 100_000, 250_000),
    ("Data Scientist", "Python;sklearn;pandas;SQL;Jupyter", 120_000, 300_000),
    ("ML Engineer", "Python;PyTorch;Kubernetes;MLflow;Airflow", 180_000, 400_000),
    ("Backend Developer", "Python;Django;Redis;PostgreSQL;Linux", 90_000, 200_000),
    ("Data Engineer", "Python;Spark;Airflow;Kafka;SQL", 140_000, 320_000),
    ("Data Analyst", "SQL;Python;Tableau;Excel;Power BI", 70_000, 160_000),
    ("DevOps Engineer", "Docker;Kubernetes;Terraform;Jenkins;Linux", 150_000, 300_000),
    ("Frontend Developer", "JavaScript;React;TypeScript;CSS;HTML", 80_000, 180_000),
    ("MLOps Engineer", "Python;Docker;Kubernetes;MLflow;Airflow", 170_000, 350_000),
    ("NLP Engineer", "Python;BERT;HuggingFace;PyTorch;spaCy", 160_000, 320_000),
    ("Senior Python Developer", "Python;FastAPI;PostgreSQL;Docker;Kubernetes;Redis", 200_000, 400_000),
    ("Junior Data Scientist", "Python;pandas;sklearn;numpy;SQL", 60_000, 120_000),
    ("Lead ML Engineer", "Python;PyTorch;TensorFlow;MLflow;Kubernetes", 300_000, 500_000),
    ("Java Developer", "Java;Spring;Kafka;PostgreSQL;Docker", 120_000, 280_000),
    ("Go Developer", "Go;gRPC;PostgreSQL;Redis;Kubernetes", 150_000, 320_000),
    ("Business Analyst", "SQL;Excel;Tableau;Power BI;BPMN", 80_000, 170_000),
    ("Product Analyst", "SQL;Python;Amplitude;Mixpanel;A/B testing", 90_000, 190_000),
    ("Computer Vision Engineer", "Python;OpenCV;PyTorch;CUDA;TensorFlow", 180_000, 380_000),
    ("Android Developer", "Kotlin;Android;Java;Git;REST API", 100_000, 220_000),
    ("iOS Developer", "Swift;Objective-C;Xcode;Git;REST API", 110_000, 230_000),
]

PREFIXES = ["Senior ", "Junior ", "Middle ", "Lead ", "Staff ", ""]
EXPERIENCES = ["noExperience", "between1And3", "between3And6", "moreThan6"]
EXP_WEIGHTS = [0.15, 0.35, 0.35, 0.15]
EMPLOYMENTS = ["full", "part", "project", "volunteer"]
EMP_WEIGHTS = [0.75, 0.1, 0.1, 0.05]
SCHEDULES = ["fullDay", "remote", "flexible", "shift"]
SCHED_WEIGHTS = [0.5, 0.3, 0.15, 0.05]
CURRENCIES = ["RUR", "USD", "EUR"]
CURR_WEIGHTS = [0.85, 0.1, 0.05]

CITIES = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Челябинск", "Самара", "Уфа", "Ростов-на-Дону",
    "Пермь", "Воронеж", "Волгоград", "Краснодар", "Саратов",
]
CITY_WEIGHTS = [0.35, 0.2, 0.07, 0.07, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03,
                0.02, 0.02, 0.02, 0.02, 0.01]

COMPANIES = [
    "Яндекс", "Сбер", "VK", "Тинькофф", "Озон", "Авито", "Mail.ru",
    "Газпромнефть", "Ростелеком", "Wildberries", "ИТМО", "МТС", "Мегафон",
    "Альфа-Банк", "ВТБ", "Раифайзен", "Лаборатория Касперского", "1С", "EPAM", "Luxoft",
    "Сколтех", "Акрос", "Evrone", "Wrike", "JetBrains", "Positive Technologies",
    "SKB Kontur", "2ГИС", "ГК Иннополис", "DataArt",
]

DESCRIPTIONS = [
    "Разработка backend сервисов на Python с использованием современных фреймворков",
    "Создание и поддержка ML моделей для продуктовых задач компании",
    "Анализ больших данных и построение дата-пайплайнов для бизнеса",
    "Разработка и оптимизация алгоритмов обработки данных в режиме реального времени",
    "Проектирование архитектуры микросервисов и интеграций с внешними системами",
    "Исследование и внедрение новых подходов в области машинного обучения",
    "Построение аналитических дашбордов и отчётности для принятия бизнес-решений",
    "Разработка NLP-систем для обработки текстов на русском и английском языках",
    "Создание рекомендательных систем на основе коллаборативной фильтрации",
    "Сопровождение и развитие MLOps платформы для команды data science",
    "Внедрение CI/CD процессов и автоматизация деплоя моделей в продакшн",
    "Разработка API и интеграция с внешними сервисами и базами данных",
    "Работа с большими объёмами данных в Spark и Hadoop кластерах",
    "Участие в исследованиях в области компьютерного зрения и обработки изображений",
    "Разработка мобильных приложений с использованием современных технологий",
]

EXP_SALARY_MULT = {
    "noExperience": 0.65,
    "between1And3": 0.85,
    "between3And6": 1.0,
    "moreThan6": 1.3,
}

rows = []
for i in range(N):
    pos_idx = np.random.randint(len(POSITIONS))
    pos_name, pos_skills, base_from, base_to = POSITIONS[pos_idx]

    prefix = np.random.choice(PREFIXES, p=[0.15, 0.15, 0.2, 0.1, 0.05, 0.35])
    if prefix and not any(p in pos_name for p in ["Senior", "Junior", "Lead"]):
        name = prefix + pos_name
    else:
        name = pos_name

    experience = np.random.choice(EXPERIENCES, p=EXP_WEIGHTS)
    mult = EXP_SALARY_MULT[experience]

    sal_from = int(base_from * mult * np.random.uniform(0.85, 1.15) / 1000) * 1000
    sal_to = int(base_to * mult * np.random.uniform(0.85, 1.15) / 1000) * 1000
    if sal_to < sal_from:
        sal_to = sal_from + 30_000

    currency = np.random.choice(CURRENCIES, p=CURR_WEIGHTS)
    if currency == "USD":
        sal_from = int(sal_from / 90 / 100) * 100
        sal_to = int(sal_to / 90 / 100) * 100
    elif currency == "EUR":
        sal_from = int(sal_from / 100 / 100) * 100
        sal_to = int(sal_to / 100 / 100) * 100

    hide_from = np.random.random() < 0.1
    hide_to = np.random.random() < 0.1

    extra_skills = np.random.choice(
        ["Git", "Linux", "Agile", "Scrum", "REST API", "gRPC", "GraphQL", "MongoDB", "Elasticsearch"],
        size=np.random.randint(0, 3), replace=False
    )
    skills = pos_skills + (";" + ";".join(extra_skills) if len(extra_skills) > 0 else "")

    rows.append({
        "id": i + 1,
        "name": name,
        "salary_from": None if hide_from else sal_from,
        "salary_to": None if hide_to else sal_to,
        "salary_currency": currency,
        "experience": experience,
        "employment": np.random.choice(EMPLOYMENTS, p=EMP_WEIGHTS),
        "schedule": np.random.choice(SCHEDULES, p=SCHED_WEIGHTS),
        "city": np.random.choice(CITIES, p=CITY_WEIGHTS),
        "company": np.random.choice(COMPANIES),
        "description": np.random.choice(DESCRIPTIONS),
        "skills": skills,
        "url": f"https://hh.ru/vacancy/{i + 1}",
    })

df = pd.DataFrame(rows)
df.to_csv("data/vacancies.csv", index=False)
print(f"Сгенерировано {len(df)} строк → data/vacancies.csv")
print(df.dtypes)
print(df.head(3))
