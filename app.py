"""
app.py — Streamlit demo for Student Performance Prediction.

Run locally:
    streamlit run app.py

Or deploy for free on Streamlit Community Cloud (streamlit.io/cloud) by
pointing it at this repo — that gives you a public demo link for your
resume/portfolio.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_student

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Student Performance Predictor")
st.caption(
    "Predicts whether a student will fall into the **Low / Medium / High** final-grade "
    "band, trained on the UCI Student Performance dataset (Cortez & Silva, 2008)."
)

st.divider()

with st.form("student_form"):
    st.subheader("Academic history")
    c1, c2, c3 = st.columns(3)
    G1 = c1.slider("Period 1 grade (G1)", 0, 20, 12)
    G2 = c2.slider("Period 2 grade (G2)", 0, 20, 12)
    failures = c3.selectbox("Past class failures", [0, 1, 2, 3], index=0)

    st.subheader("Study habits & attendance")
    c1, c2, c3 = st.columns(3)
    studytime = c1.selectbox(
        "Weekly study time", [1, 2, 3, 4],
        format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x],
        index=1,
    )
    absences = c2.slider("Absences (school year)", 0, 75, 4)
    higher = c3.selectbox("Wants higher education?", ["yes", "no"], index=0)

    st.subheader("Lifestyle")
    c1, c2, c3 = st.columns(3)
    goout = c1.slider("Going out with friends (1=low, 5=high)", 1, 5, 3)
    Dalc = c2.slider("Workday alcohol use (1-5)", 1, 5, 1)
    Walc = c3.slider("Weekend alcohol use (1-5)", 1, 5, 2)

    st.subheader("Background")
    c1, c2, c3 = st.columns(3)
    school = c1.selectbox("School", ["GP", "MS"])
    sex = c2.selectbox("Sex", ["F", "M"])
    address = c3.selectbox("Home address", ["U", "R"], format_func=lambda x: "Urban" if x == "U" else "Rural")

    c1, c2 = st.columns(2)
    Medu = c1.selectbox("Mother's education (0=none, 4=higher)", [0, 1, 2, 3, 4], index=2)
    Fedu = c2.selectbox("Father's education (0=none, 4=higher)", [0, 1, 2, 3, 4], index=2)

    with st.expander("Advanced (rarely needs changing)"):
        famsize = st.selectbox("Family size", ["LE3", "GT3"], index=1)
        Pstatus = st.selectbox("Parents' cohabitation", ["T", "A"], index=0)
        Mjob = st.selectbox("Mother's job", ["teacher", "health", "services", "at_home", "other"], index=2)
        Fjob = st.selectbox("Father's job", ["teacher", "health", "services", "at_home", "other"], index=4)
        reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"], index=2)
        guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=0)
        traveltime = st.selectbox("Home-to-school travel time", [1, 2, 3, 4], index=0)
        schoolsup = st.selectbox("Extra school support", ["yes", "no"], index=1)
        famsup = st.selectbox("Family educational support", ["yes", "no"], index=0)
        paid = st.selectbox("Extra paid classes", ["yes", "no"], index=1)
        activities = st.selectbox("Extracurricular activities", ["yes", "no"], index=0)
        nursery = st.selectbox("Attended nursery school", ["yes", "no"], index=0)
        internet = st.selectbox("Internet access at home", ["yes", "no"], index=0)
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"], index=1)
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        health = st.slider("Health status (1-5)", 1, 5, 3)
        age = st.slider("Age", 15, 22, 17)

    submitted = st.form_submit_button("Predict performance", use_container_width=True)

if submitted:
    student = {
        "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
        "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
        "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
        "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
        "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences, "G1": G1, "G2": G2,
    }

    result = predict_student(student)
    pred = result["prediction"]

    st.divider()
    color = {"High": "green", "Medium": "orange", "Low": "red"}.get(pred, "blue")
    st.markdown(f"### Predicted performance band: :{color}[{pred}]")

    if result["probabilities"]:
        proba_df = pd.DataFrame(
            {"Class": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
        ).sort_values("Probability", ascending=False)
        st.bar_chart(proba_df.set_index("Class"))

st.divider()
st.caption(
    "Model: tuned XGBoost classifier (5-fold CV macro-F1 ≈ 0.90). "
    "Source & full report: see the GitHub repository README."
)
