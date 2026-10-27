import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import fuzzy_model, ml_models, reports, whatif
from modules.config import *

st.set_page_config(page_title="Оценка инвестпроектов", layout="wide")
st.title("Оценка инвестиционных проектов")

if not (os.path.exists(FINAL_PATH) and os.path.exists(BEST_MODEL_PATH)):
    st.warning("Сначала запустите: python main.py --all")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(FINAL_PATH, encoding="utf-8-sig")


@st.cache_resource
def load_model():
    return ml_models.load_best()


def show_image(name, caption=None):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption)


def project_inputs(prefix, defaults):
    project = {"sector": st.selectbox("Сектор", SECTORS, index=SECTORS.index(defaults["sector"]), key=prefix + "sector")}
    cols = st.columns(3)
    for i, col in enumerate(NUM_COLS):
        lo, hi = RANGES[col]
        value = float(defaults[col])
        with cols[i % 3]:
            if col in INT_COLS:
                project[col] = st.number_input(col, int(lo), int(hi), int(value), step=1, key=prefix + col)
            else:
                project[col] = st.number_input(col, float(lo), float(hi), min(max(value, lo), hi), key=prefix + col)
    return project


def show_result(r):
    c = st.columns(5)
    c[0].metric("Финансовый", round(r["fin_risk"], 1))
    c[1].metric("Рыночный", round(r["mkt_risk"], 1))
    c[2].metric("Организационный", round(r["org_risk"], 1))
    c[3].metric("fuzzy_risk", round(r["fuzzy_risk"], 1))
    c[4].metric("Класс", CLASS_NAMES[r["class"]])
    st.bar_chart(pd.DataFrame({"вероятность": r["proba"]}, index=[CLASS_NAMES[i] for i in range(3)]))


df = load_data()
bundle = load_model()
medians = df[NUM_COLS].median().to_dict()
medians["sector"] = SECTORS[0]

tabs = st.tabs(["Данные", "Нечеткая модель", "Метрики", "SHAP", "What-if", "Новый проект"])

with tabs[0]:
    st.dataframe(df.head(200))
    st.write(df[NUM_COLS + DERIVED_COLS].describe().round(2))
    st.bar_chart(df["risk_class"].value_counts().sort_index())
    show_image("eda_corr.png")
    show_image("eda_boxplots.png")
    show_image("eda_hist.png")

with tabs[1]:
    st.subheader("Правила")
    rules = fuzzy_model.rules_table()
    level = st.radio("Уровень", [1, 2], horizontal=True)
    st.dataframe(rules[rules["level"] == level], width="stretch")
    st.subheader("Поверхность fuzzy_risk")
    _, (xs, ys, z) = fuzzy_model.draw_surface(df)
    fig = go.Figure(data=[go.Surface(x=xs, y=ys, z=z)])
    fig.update_layout(scene=dict(xaxis_title="profitability", yaxis_title="payback_period", zaxis_title="fuzzy_risk"),
                      height=500)
    st.plotly_chart(fig, width="stretch")
    show_image("fuzzy_sensitivity.png")
    st.subheader("Трассировка правил")
    pid = st.selectbox("project_id", df["project_id"].head(300))
    row = df[df["project_id"] == pid].iloc[0]
    table, values = fuzzy_model.trace_rules(row.to_dict())
    st.write({k: round(v, 1) for k, v in values.items()})
    st.dataframe(table, width="stretch")

with tabs[2]:
    if os.path.exists(COMPARISON_PATH):
        st.dataframe(pd.read_csv(COMPARISON_PATH, encoding="utf-8-sig").round(3))
    show_image("confusion_matrices.png")
    biz = os.path.join(RES_DIR, "business_models.csv")
    if os.path.exists(biz):
        st.subheader("Бизнес: ожидаемые потери")
        st.dataframe(pd.read_csv(biz, encoding="utf-8-sig").round(3))
        show_image("business_loss.png")
        text_path = os.path.join(RES_DIR, "business_conclusion.txt")
        if os.path.exists(text_path):
            st.text(open(text_path, encoding="utf-8").read())

with tabs[3]:
    show_image("shap_summary.png")
    show_image("shap_waterfall.png")
    show_image("shap_classes.png")

with tabs[4]:
    pid = st.selectbox("Проект", df["project_id"].head(300), key="wi_pid")
    row = df[df["project_id"] == pid].iloc[0]
    project = {"sector": row["sector"]}
    for col in NUM_COLS:
        project[col] = row[col]
    chosen = st.multiselect("Что менять (1-2 параметра)", NUM_COLS, default=["payback_period"], max_selections=2)
    changes = {}
    for col in chosen:
        lo, hi = RANGES[col]
        value = st.slider(col, float(lo), float(hi), float(min(max(project[col], lo), hi)), key="wi_" + col)
        changes[col] = int(round(value)) if col in INT_COLS else value
    before, after = whatif.what_if(bundle, project, changes)
    left, right = st.columns(2)
    with left:
        st.subheader("До")
        show_result(before)
    with right:
        st.subheader("После")
        show_result(after)
    if st.button("Монте-Карло (500 прогонов)"):
        summary, runs = whatif.monte_carlo(bundle, project, N_MC)
        st.write({"устойчивость класса": round(summary["stability"], 3),
                  "fuzzy среднее": round(summary["fuzzy_mean"], 1),
                  "fuzzy std": round(summary["fuzzy_std"], 1)})
        st.bar_chart(runs["fuzzy_risk"].round(0).value_counts().sort_index())

with tabs[5]:
    new_project = project_inputs("new_", medians)
    if st.button("Оценить проект"):
        result = whatif.evaluate_project(bundle, new_project)
        show_result(result)
        st.info(reports.recommend(result["class"], result["fin_risk"], result["mkt_risk"], result["org_risk"]))
        table, _ = fuzzy_model.trace_rules(whatif.prepare(pd.DataFrame([new_project])).iloc[0].to_dict())
        st.dataframe(table, width="stretch")
