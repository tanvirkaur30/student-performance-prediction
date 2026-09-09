"""
predict.py
----------
Loads the trained pipeline (models/best_model.pkl) and target encoder,
and exposes a `predict_student(student_dict)` function used by both
app.py (Streamlit demo) and this CLI.

CLI usage:
    python src/predict.py --school GP --sex F --age 17 --address U ...
    (or simply run with no args to see a demo prediction on a sample student)
"""

import argparse
import os

import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "target_encoder.pkl")

_model = None
_encoder = None


def _load():
    global _model, _encoder
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _encoder is None:
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _encoder


def _engineer(student: dict) -> dict:
    """Apply the same feature engineering used in training (preprocess.add_features)."""
    student = dict(student)
    student["avg_alcohol"] = (student["Dalc"] + student["Walc"]) / 2
    student["parent_edu_avg"] = (student["Medu"] + student["Fedu"]) / 2
    student["avg_prior_grade"] = (student["G1"] + student["G2"]) / 2
    return student


def predict_student(student: dict) -> dict:
    """
    Predict a student's performance category.

    Parameters
    ----------
    student : dict
        Raw feature values using the ORIGINAL UCI column names/values,
        e.g. {'school': 'GP', 'sex': 'F', 'age': 17, 'address': 'U', ...}

    Returns
    -------
    dict with keys: 'prediction' (str) and 'probabilities' (dict of class->prob)
    """
    model, encoder = _load()
    student = _engineer(student)
    X = pd.DataFrame([student])

    pred_encoded = model.predict(X)[0]
    prediction = encoder.inverse_transform([pred_encoded])[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = encoder.inverse_transform(range(len(proba)))
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    return {"prediction": prediction, "probabilities": probabilities}


DEFAULT_SAMPLE = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 3, "Fedu": 2, "Mjob": "services", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
    "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1,
    "Walc": 2, "health": 3, "absences": 4, "G1": 14, "G2": 15,
}


def main():
    parser = argparse.ArgumentParser(description="Predict a student's performance category.")
    for key, default in DEFAULT_SAMPLE.items():
        parser.add_argument(f"--{key}", default=default, type=type(default))
    args = parser.parse_args()

    student = {key: getattr(args, key) for key in DEFAULT_SAMPLE}
    result = predict_student(student)
    print("Input:", student)
    print("\nPredicted performance category:", result["prediction"])
    print("Class probabilities:", result["probabilities"])


if __name__ == "__main__":
    main()
