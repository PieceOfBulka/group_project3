import os
import json
import threading
import queue
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ML Агент — Зарплаты HH.ru", page_icon="🤖", layout="wide")

st.title("🤖 ML Агент: Предсказание зарплат HH.ru")
st.caption("Агент автономно парсит данные, обучает модели и предсказывает зарплату")

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("OpenRouter API Key", value=os.getenv("API_KEY", ""), type="password")
    model_name = st.selectbox(
        "LLM модель",
        [
            "minimax/minimax-m2.5:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.3-8b-instruct:free",
        ],
        index=0,
    )
    st.divider()
    st.header("🎯 Вакансия для предсказания")
    vac_name = st.text_input("Название", value="Middle Python Developer")
    vac_exp = st.selectbox(
        "Опыт",
        ["noExperience", "between1And3", "between3And6", "moreThan6"],
        index=1,
    )
    vac_city = st.selectbox(
        "Город",
        ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", "Другой"],
        index=0,
    )
    vac_schedule = st.selectbox("График", ["fullDay", "remote", "flexible"], index=0)
    vac_skills = st.text_input("Навыки (через ;)", value="Python;FastAPI;PostgreSQL;Docker;Git")

STEPS = [
    ("parse_hh_vacancies", "Парсинг HH.ru"),
    ("load_and_explore_data", "Загрузка и EDA"),
    ("preprocess_data", "Предобработка + Feature Engineering"),
    ("train_and_compare_models", "Обучение моделей"),
    ("predict_salary", "Предсказание зарплаты"),
    ("generate_report", "Генерация отчёта"),
]

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "running" not in st.session_state:
    st.session_state.running = False
if "done" not in st.session_state:
    st.session_state.done = False
if "result" not in st.session_state:
    st.session_state.result = None
if "report_path" not in st.session_state:
    st.session_state.report_path = None
if "step_status" not in st.session_state:
    st.session_state.step_status = {k: "pending" for k, _ in STEPS}


col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 Прогресс")
    step_placeholders = {}
    for key, label in STEPS:
        step_placeholders[key] = st.empty()

    def render_steps():
        for key, label in STEPS:
            status = st.session_state.step_status.get(key, "pending")
            icon = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌"}[status]
            step_placeholders[key].markdown(f"{icon} {label}")

    render_steps()

with col1:
    run_btn = st.button(
        "🚀 Запустить агента",
        disabled=st.session_state.running,
        type="primary",
        use_container_width=True,
    )

    log_box = st.empty()

    def render_log():
        text = "\n".join(st.session_state.log_lines[-80:])
        log_box.code(text, language=None)

    render_log()

    if st.session_state.result:
        st.divider()
        st.subheader("📊 Финальный ответ агента")
        st.markdown(st.session_state.result)

    if st.session_state.report_path and os.path.exists(st.session_state.report_path):
        with open(st.session_state.report_path, encoding="utf-8") as f:
            html_content = f.read()
        st.divider()
        st.subheader("📈 HTML-отчёт")
        st.components.v1.html(html_content, height=600, scrolling=True)


if run_btn:
    if not api_key:
        st.error("Введи API Key в боковой панели")
        st.stop()

    os.environ["API_KEY"] = api_key
    os.environ["MODEL_NAME"] = model_name

    target_vacancy = {
        "name": vac_name,
        "experience": vac_exp,
        "employment": "full",
        "schedule": vac_schedule,
        "city": vac_city,
        "skills": vac_skills,
    }

    st.session_state.log_lines = []
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.result = None
    st.session_state.report_path = None
    st.session_state.step_status = {k: "pending" for k, _ in STEPS}

    log_queue = queue.Queue()

    class QueueWriter:
        def write(self, text):
            if text.strip():
                log_queue.put(text.rstrip())
        def flush(self):
            pass

    import sys

    def run_agent():
        old_stdout = sys.stdout
        sys.stdout = QueueWriter()
        try:
            from tools.state import STATE
            STATE["action_history"] = []
            STATE["df_processed"] = None
            STATE["feature_cols"] = None
            STATE["best_model"] = None
            STATE["best_model_name"] = None
            STATE["model_results"] = None

            from agent import run
            final = run(
                csv_filepath="data/vacancies.csv",
                target_vacancy=target_vacancy,
                model_name=model_name,
            )
            log_queue.put(f"__DONE__{json.dumps({'result': final}, ensure_ascii=False)}")
        except Exception as e:
            log_queue.put(f"__ERROR__{e}")
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    STEP_KEYWORDS = {
        "parse_hh_vacancies": ["Парсинг завершён", "parse_hh", "Подключаюсь к HH"],
        "load_and_explore_data": ["load_and_explore", "КОД (попытка", "Загружено"],
        "preprocess_data": ["preprocess_data", "Обработано"],
        "train_and_compare_models": ["train_and_compare", "Лучшая:"],
        "predict_salary": ["predict_salary", "predicted_salary"],
        "generate_report": ["generate_report", "report_"],
    }

    current_step_idx = 0

    while thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.3)
        except queue.Empty:
            render_log()
            render_steps()
            continue

        if line.startswith("__DONE__"):
            data = json.loads(line[8:])
            st.session_state.result = data.get("result", "")
            st.session_state.running = False
            st.session_state.done = True
            for key, _ in STEPS:
                st.session_state.step_status[key] = "done"

            from tools.state import STATE as _STATE
            history = _STATE.get("action_history", [])
            for h in history:
                if h["tool"] == "generate_report" and "report_path" in h.get("summary", ""):
                    pass
            import glob
            reports = sorted(glob.glob("reports/report_*.html"))
            if reports:
                st.session_state.report_path = reports[-1]
            break

        elif line.startswith("__ERROR__"):
            st.session_state.log_lines.append(f"❌ ОШИБКА: {line[9:]}")
            st.session_state.running = False
            for key, _ in STEPS:
                if st.session_state.step_status[key] == "running":
                    st.session_state.step_status[key] = "error"
            break

        else:
            st.session_state.log_lines.append(line)

            for key, _ in STEPS:
                for kw in STEP_KEYWORDS.get(key, []):
                    if kw in line:
                        if st.session_state.step_status[key] == "pending":
                            st.session_state.step_status[key] = "running"
                        elif st.session_state.step_status[key] == "running":
                            st.session_state.step_status[key] = "done"

            tool_line = ""
            if "Tool:" in line:
                tool_line = line.split("Tool:")[-1].split("(")[0].strip()
                for key, _ in STEPS:
                    if key in tool_line:
                        if st.session_state.step_status[key] == "pending":
                            st.session_state.step_status[key] = "running"

        render_log()
        render_steps()

    render_log()
    render_steps()
    st.rerun()
