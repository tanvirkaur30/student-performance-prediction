"""
preprocess.py
-------------
Data loading, feature engineering, and preprocessing pipeline
for the Student Performance Prediction project.

Author: Dhruv Kumar, Tanvir Kaur, Farhan Kazi
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
BINARY_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NOMINAL_COLS = ["Mjob", "Fjob", "reason", "guardian"]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
    # engineered features (added in add_features)
    "avg_alcohol", "parent_edu_avg", "avg_prior_grade",
]

TARGET_RAW = "G3"
TARGET_CLASS = "G3_cat"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw UCI student-mat.csv file (comma or semicolon separated)."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features that add real predictive/explanatory value.

    - avg_alcohol: mean of weekday + weekend alcohol consumption
    - parent_edu_avg: mean of mother's and father's education level
    - avg_prior_grade: mean of G1 and G2 (used only in the "with priors" model)
    """
    df = df.copy()
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
    df["avg_prior_grade"] = (df["G1"] + df["G2"]) / 2
    return df


def add_target_class(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket the numeric G3 grade (0-20) into Low / Medium / High classes."""
    df = df.copy()
    df[TARGET_CLASS] = pd.cut(
        df[TARGET_RAW], bins=[-1, 9, 14, 20], labels=["Low", "Medium", "High"]
    )
    return df


def build_preprocessor(drop_prior_grades: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot encodes nominal features,
    label/ordinal-encodes binary yes/no features, and scales numerics.

    Parameters
    ----------
    drop_prior_grades : bool
        If True, G1/G2/avg_prior_grade are excluded — used for the
        "harder" experiment that predicts performance WITHOUT knowing
        the student's earlier grades (see report, Section: Feature
        Engineering / Ablation Study).
    """
    numeric_cols = NUMERIC_COLS.copy()
    if drop_prior_grades:
        for c in ("G1", "G2", "avg_prior_grade"):
            if c in numeric_cols:
                numeric_cols.remove(c)

    preprocessor = ColumnTransformer(
        transformers=[
            ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), BINARY_COLS),
            ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL_COLS),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return output feature names after a fitted ColumnTransformer transform."""
    return list(preprocessor.get_feature_names_out())


def prepare_dataset(path: str, drop_prior_grades: bool = False):
    """End-to-end: load -> engineer features -> bucket target -> split X/y.

    Returns
    -------
    X : pd.DataFrame (raw feature columns, NOT yet encoded/scaled)
    y : pd.Series (encoded Low=1, Medium=2, High=0 style int labels)
    target_encoder : fitted LabelEncoder for decoding predictions back to text
    """
    df = load_data(path)
    df = add_features(df)
    df = add_target_class(df)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_CLASS].astype(str))

    feature_cols = BINARY_COLS + NOMINAL_COLS + NUMERIC_COLS
    if drop_prior_grades:
        feature_cols = [c for c in feature_cols if c not in ("G1", "G2", "avg_prior_grade")]

    X = df[feature_cols]
    return X, pd.Series(y, name=TARGET_CLASS), target_encoder
