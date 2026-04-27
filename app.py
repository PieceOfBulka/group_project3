import os
import json
import sys
import glob
import threading
import queue
import csv as csv_mod
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ML Агент — Зарплаты HH.ru", page_icon="🤖", layout="wide")

# ── сайдбар ──────────────────────────────────────────────────────────────────
api_key = os.getenv("API_KEY", "")

with st.sidebar:
    st.header("⚙️ Настройки")
    model_name = st.selectbox(
        "LLM модель (агент)",
        [
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-v4-flash",
            "minimax/minimax-m2.7",
            "openai/gpt-oss-120b:free",
            "z-ai/glm-4.5-air:free",
            "google/gemma-3-27b-it:free",
            "minimax/minimax-m2.5:free",
        ],
    )
    st.divider()
    st.header("🎯 Вакансия для предсказания")
    vac_name = st.text_input("Название", value="Middle Python Developer")
    vac_exp = st.selectbox("Опыт", ["noExperience", "between1And3", "between3And6", "moreThan6"], index=1)
    vac_city = st.selectbox("Город", ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", "Другой"])
    vac_schedule = st.selectbox("График", ["fullDay", "remote", "flexible"])
    vac_skills = st.text_input("Навыки (через ;)", value="Python;FastAPI;PostgreSQL;Docker;Git")

# ── вкладки ──────────────────────────────────────────────────────────────────
tab_agent, tab_benchmark = st.tabs(["🤖 ML Агент", "📊 Сравнение LLM"])

# ═══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1: ML АГЕНТ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.title("🤖 ML Агент: Предсказание зарплат HH.ru")
    st.caption("Агент автономно парсит данные, обучает модели и предсказывает зарплату")

    STEPS = [
        ("parse_hh_vacancies", "Парсинг HH.ru (демо)"),
        ("load_and_explore_data", "Загрузка и EDA"),
        ("preprocess_data", "Предобработка + Feature Engineering"),
        ("train_and_compare_models", "Обучение моделей"),
        ("predict_salary", "Предсказание зарплаты"),
        ("generate_report", "Генерация отчёта"),
    ]

    for key in ["log_lines", "running", "done", "result", "report_path", "step_status", "model_results", "best_model_name"]:
        if key not in st.session_state:
            st.session_state[key] = (
                [] if key == "log_lines" else
                False if key in ("running", "done") else
                None if key in ("result", "report_path", "model_results", "best_model_name") else
                {k: "pending" for k, _ in STEPS}
            )

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("📋 Прогресс")
        step_placeholders = {k: st.empty() for k, _ in STEPS}

        def render_steps():
            for k, label in STEPS:
                icon = {"pending": "⬜", "running": "🔄", "done": "✅", "error": "❌"}[
                    st.session_state.step_status.get(k, "pending")
                ]
                step_placeholders[k].markdown(f"{icon} {label}")

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
            log_box.code("\n".join(st.session_state.log_lines[-80:]), language=None)

        render_log()

        if st.session_state.result:
            st.divider()
            st.subheader("📊 Финальный ответ агента")
            st.markdown(st.session_state.result)

        if st.session_state.model_results:
            st.divider()
            st.subheader("💾 Сохранение модели")

            import pickle, io
            results = st.session_state.model_results
            best_name = st.session_state.best_model_name

            df_models = pd.DataFrame(results)[["name", "mae", "rmse", "r2"]]
            df_models.columns = ["Модель", "MAE", "RMSE", "R²"]
            df_models["MAE"] = df_models["MAE"].apply(lambda x: f"{x:,.0f} ₽")
            df_models["RMSE"] = df_models["RMSE"].apply(lambda x: f"{x:,.0f} ₽")
            df_models["R²"] = df_models["R²"].apply(lambda x: f"{x:.4f}")
            df_models.insert(0, "🏆", df_models["Модель"].apply(lambda x: "✅" if x == best_name else ""))
            st.dataframe(df_models, use_container_width=True, hide_index=True)

            save_col1, save_col2 = st.columns([2, 1])
            with save_col1:
                model_to_save = st.selectbox(
                    "Выбери модель для сохранения",
                    options=[r["name"] for r in results],
                    index=next((i for i, r in enumerate(results) if r["name"] == best_name), 0),
                    key="model_select",
                )
                save_path = st.text_input("Путь для сохранения", value=f"memory/{model_to_save.replace(' ', '_').lower()}.pkl", key="save_path")

            with save_col2:
                st.write("")
                st.write("")
                if st.button("💾 Сохранить в файл", use_container_width=True):
                    from tools.state import STATE as _S2
                    model_obj = _S2.get("best_model")
                    if model_obj is not None:
                        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
                        with open(save_path, "wb") as f:
                            pickle.dump(model_obj, f)
                        st.success(f"Сохранено: `{save_path}`")
                    else:
                        st.error("Модель не найдена в памяти агента")

                from tools.state import STATE as _S3
                model_obj = _S3.get("best_model")
                if model_obj is not None:
                    buf = io.BytesIO()
                    pickle.dump(model_obj, buf)
                    buf.seek(0)
                    fname = f"{best_name.replace(' ', '_').lower()}.pkl" if best_name else "model.pkl"
                    st.download_button("⬇ Скачать .pkl", buf, file_name=fname, mime="application/octet-stream", use_container_width=True)

            pkl_files = sorted(glob.glob("memory/*.pkl"))
            if pkl_files:
                st.caption(f"📁 Сохранённые модели: {', '.join(os.path.basename(p) for p in pkl_files)}")

        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, encoding="utf-8") as f:
                html_content = f.read()
            st.divider()
            st.subheader("📈 HTML-отчёт")
            st.components.v1.html(html_content, height=600, scrolling=True)

    if run_btn:
        if not api_key:
            st.error("API_KEY не найден. Добавь его в .env файл")
            st.stop()

        os.environ["MODEL_NAME"] = model_name

        target_vacancy = {
            "name": vac_name, "experience": vac_exp, "employment": "full",
            "schedule": vac_schedule, "city": vac_city, "skills": vac_skills,
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
            def flush(self): pass

        def run_agent():
            old_stdout = sys.stdout
            sys.stdout = QueueWriter()
            try:
                from tools.state import STATE
                STATE.update({"action_history": [], "df_processed": None, "feature_cols": None,
                               "best_model": None, "best_model_name": None, "model_results": None})
                _STATE_REF = STATE
                from agent import run
                final = run(csv_filepath="data/hh_it_5000_final.csv", target_vacancy=target_vacancy, model_name=model_name)
                log_queue.put(f"__DONE__{json.dumps({'result': final}, ensure_ascii=False)}")
            except Exception as e:
                log_queue.put(f"__ERROR__{e}")
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        st.session_state.step_status["parse_hh_vacancies"] = "done"
        render_steps()

        STEP_KEYWORDS = {
            "parse_hh_vacancies": [],
            "load_and_explore_data": ["Загружено", "load_and_explore"],
            "preprocess_data": ["Обработано", "preprocess_data"],
            "train_and_compare_models": ["Лучшая:", "train_and_compare"],
            "predict_salary": ["predict_salary", "predicted_salary"],
            "generate_report": ["generate_report", "report_"],
        }

        while thread.is_alive() or not log_queue.empty():
            try:
                line = log_queue.get(timeout=0.3)
            except queue.Empty:
                render_log(); render_steps(); continue

            if line.startswith("__DONE__"):
                data = json.loads(line[8:])
                st.session_state.result = data.get("result", "")
                st.session_state.running = False
                st.session_state.done = True
                for k, _ in STEPS:
                    st.session_state.step_status[k] = "done"
                reports = sorted(glob.glob("reports/report_*.html"))
                if reports:
                    st.session_state.report_path = reports[-1]
                from tools.state import STATE as _S
                st.session_state.model_results = _S.get("model_results")
                st.session_state.best_model_name = _S.get("best_model_name")
                break
            elif line.startswith("__ERROR__"):
                st.session_state.log_lines.append(f"❌ ОШИБКА: {line[9:]}")
                st.session_state.running = False
                for k, _ in STEPS:
                    if st.session_state.step_status[k] == "running":
                        st.session_state.step_status[k] = "error"
                break
            else:
                st.session_state.log_lines.append(line)
                for k, _ in STEPS:
                    for kw in STEP_KEYWORDS.get(k, []):
                        if kw in line:
                            cur = st.session_state.step_status[k]
                            if cur == "pending":
                                st.session_state.step_status[k] = "running"
                            elif cur == "running":
                                st.session_state.step_status[k] = "done"
                if "Tool:" in line:
                    tool_name = line.split("Tool:")[-1].split("(")[0].strip()
                    for k, _ in STEPS:
                        if k in tool_name and st.session_state.step_status[k] == "pending":
                            st.session_state.step_status[k] = "running"

            render_log()
            render_steps()

        render_log()
        render_steps()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2: СРАВНЕНИЕ LLM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.title("📊 Сравнение LLM-моделей")
    st.caption("Каждая модель проходит 3 задачи: EDA, предобработка, обучение моделей")

    BENCH_MODELS = [
        {"name": "GPT-OSS 120B", "id": "openai/gpt-oss-120b:free", "tier": "🆓 free"},
        {"name": "GLM-4.5 Air", "id": "z-ai/glm-4.5-air:free", "tier": "🆓 free"},
        {"name": "DeepSeek V4 Flash", "id": "deepseek/deepseek-v4-flash", "tier": "💰 paid"},
        {"name": "MiniMax M2.7", "id": "minimax/minimax-m2.7", "tier": "💰 paid"},
    ]

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        selected_models = st.multiselect(
            "Модели для тестирования",
            options=[m["id"] for m in BENCH_MODELS],
            default=[m["id"] for m in BENCH_MODELS],
            format_func=lambda x: next(m["name"] for m in BENCH_MODELS if m["id"] == x),
        )
        run_bench = st.button("▶ Запустить бенчмарк", type="primary", use_container_width=True)

    with b_col2:
        st.info(
            "**Метрики оценки:**\n"
            "- ✅ tasks_passed — сколько из 3 задач выполнено без ошибки\n"
            "- 🔁 total_retries — сколько раз LLM пришлось исправлять код\n"
            "- ⏱ total_time_sec — суммарное время генерации кода\n"
            "- 📉 best_mae / best_r2 — качество обученной ML-модели\n"
            "- 💯 success_rate_pct — процент успешных задач"
        )

    CSV_PATH = "llm_comparison.csv"

    if run_bench:
        if not api_key:
            st.error("API_KEY не найден. Добавь его в .env файл")
        else:
            progress = st.progress(0)
            status_txt = st.empty()
            results_placeholder = st.empty()

            from benchmark_llms import benchmark_model, FIELDNAMES

            models_to_run = [m for m in BENCH_MODELS if m["id"] in selected_models]
            all_rows = []

            for i, model_info in enumerate(models_to_run):
                status_txt.markdown(f"⏳ Тестирую **{model_info['name']}** ({i+1}/{len(models_to_run)})...")
                try:
                    row = benchmark_model(model_info)
                except Exception as e:
                    row = {"model_name": model_info["name"], "model_id": model_info["id"],
                           "tier": model_info["tier"], "error": str(e), "tasks_passed": 0, "success_rate_pct": 0}
                all_rows.append(row)
                progress.progress((i + 1) / len(models_to_run))
                results_placeholder.dataframe(pd.DataFrame(all_rows), use_container_width=True)

            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv_mod.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)

            status_txt.success(f"✅ Готово! Результаты сохранены в `{CSV_PATH}`")

    if os.path.exists(CSV_PATH):
        st.divider()
        st.subheader("📋 Результаты последнего бенчмарка")
        df_res = pd.read_csv(CSV_PATH)

        display_cols = [c for c in [
            "model_name", "tier", "tasks_passed", "success_rate_pct",
            "best_mae", "best_r2", "total_retries", "total_time_sec",
            "eda_ok", "preprocess_ok", "train_ok",
        ] if c in df_res.columns]
        st.dataframe(df_res[display_cols], use_container_width=True)

        if "best_mae" in df_res.columns and df_res["best_mae"].notna().any():
            st.subheader("📈 Сравнение метрик")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.bar_chart(df_res.set_index("model_name")["best_mae"].dropna(), y_label="MAE (меньше = лучше)")
            with c2:
                st.bar_chart(df_res.set_index("model_name")["best_r2"].dropna(), y_label="R² (больше = лучше)")
            with c3:
                st.bar_chart(df_res.set_index("model_name")["total_retries"].dropna(), y_label="Retries (меньше = лучше)")

        with open(CSV_PATH, "rb") as f:
            st.download_button("⬇ Скачать CSV", f, file_name="llm_comparison.csv", mime="text/csv")
