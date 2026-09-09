"""
train.py
--------
Trains and compares Logistic Regression, Decision Tree, Random Forest,
k-NN and XGBoost on the UCI Student Performance dataset, using a proper
sklearn Pipeline, stratified 5-fold cross-validation and hyperparameter
tuning (GridSearchCV) for the top candidates.

Saves:
    models/best_model.pkl        -> best full pipeline (preprocessing + model)
    models/target_encoder.pkl    -> label encoder for Low/Medium/High
    models/results.json          -> metrics for every model (for the README/report)
    assets/*.png                 -> confusion matrix + SHAP plots

Run:
    python src/train.py
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build_preprocessor, prepare_dataset

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def get_candidate_models():
    """Base (un-tuned) candidates for the first comparison pass."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "k-NN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0
        ),
    }


def evaluate_cv(pipeline, X, y):
    """Return mean accuracy and mean macro-F1 across 5-fold stratified CV."""
    acc = cross_val_score(pipeline, X, y, cv=CV, scoring="accuracy")
    f1 = cross_val_score(pipeline, X, y, cv=CV, scoring="f1_macro")
    return acc.mean(), acc.std(), f1.mean(), f1.std()


def tune_random_forest(preprocessor, X_train, y_train):
    """GridSearchCV over Random Forest hyperparameters."""
    pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    param_grid = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_split": [2, 4, 8],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_xgboost(preprocessor, X_train, y_train):
    """GridSearchCV over XGBoost hyperparameters."""
    pipe = Pipeline([("prep", preprocessor), ("clf", XGBClassifier(
        eval_metric="mlogloss", random_state=RANDOM_STATE, verbosity=0
    ))])
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.05, 0.1, 0.2],
    }
    search = GridSearchCV(pipe, param_grid, cv=CV, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def plot_confusion_matrix(y_true, y_pred, class_names, title, out_path):
    cm_disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", colorbar=True
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(results_df, out_path):
    plt.figure(figsize=(8, 5))
    order = results_df.sort_values("cv_f1_macro_mean", ascending=False)
    sns.barplot(data=order, x="cv_f1_macro_mean", y="model", palette="viridis")
    plt.xlabel("Mean CV Macro F1-Score (5-fold)")
    plt.ylabel("")
    plt.title("Model Comparison — Cross-Validated Macro F1")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_shap_summary(best_pipeline, X_train, feature_names, out_path):
    """SHAP summary plot for the best model's underlying classifier.

    Falls back gracefully if the model type isn't SHAP-friendly with
    TreeExplainer (e.g. it's Logistic Regression -> use LinearExplainer).
    """
    import shap

    clf = best_pipeline.named_steps["clf"]
    prep = best_pipeline.named_steps["prep"]
    X_train_t = prep.transform(X_train)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()

    sample = X_train_t[: min(150, X_train_t.shape[0])]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
    except Exception:
        explainer = shap.Explainer(clf, sample)
        shap_values = explainer(sample).values

    shap_values = np.asarray(shap_values)

    # Multiclass models return shape (n_samples, n_features, n_classes) or a
    # list of per-class arrays. Collapse to a single (n_samples, n_features)
    # "overall importance" view by averaging |SHAP value| across classes —
    # this is the standard way to summarise multiclass SHAP in one plot.
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.stack(shap_values, axis=-1)), axis=-1)
    elif shap_values.ndim == 3:
        shap_values = np.mean(np.abs(shap_values), axis=-1)

    plt.figure()
    try:
        shap.summary_plot(
            shap_values, sample, feature_names=list(feature_names),
            show=False, max_display=15, plot_type="dot",
        )
    except Exception as e:
        print("SHAP plotting skipped:", e)
        plt.close()
        return
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading and preparing data...")
    X, y, target_encoder = prepare_dataset(DATA_PATH)
    class_names = target_encoder.classes_.tolist()
    print(f"Dataset: {X.shape[0]} students, {X.shape[1]} raw features, classes={class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    # --- Pass 1: baseline comparison of all candidate models via CV ---
    print("\n=== Baseline 5-fold CV comparison ===")
    rows = []
    for name, model in get_candidate_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        acc_m, acc_s, f1_m, f1_s = evaluate_cv(pipe, X_train, y_train)
        rows.append({
            "model": name,
            "cv_accuracy_mean": round(acc_m, 4),
            "cv_accuracy_std": round(acc_s, 4),
            "cv_f1_macro_mean": round(f1_m, 4),
            "cv_f1_macro_std": round(f1_s, 4),
        })
        print(f"{name:22s} | CV Acc: {acc_m:.3f} ± {acc_s:.3f} | CV F1(macro): {f1_m:.3f} ± {f1_s:.3f}")

    results_df = pd.DataFrame(rows)
    plot_model_comparison(results_df, os.path.join(ASSET_DIR, "model_comparison.png"))

    # --- Pass 2: hyperparameter tuning on the two strongest candidates ---
    print("\n=== Hyperparameter tuning: Random Forest ===")
    rf_search = tune_random_forest(preprocessor, X_train, y_train)
    print("Best RF params:", rf_search.best_params_)
    print("Best RF CV F1(macro):", round(rf_search.best_score_, 4))

    print("\n=== Hyperparameter tuning: XGBoost ===")
    xgb_search = tune_xgboost(preprocessor, X_train, y_train)
    print("Best XGB params:", xgb_search.best_params_)
    print("Best XGB CV F1(macro):", round(xgb_search.best_score_, 4))

    candidates = {
        "Random Forest (tuned)": rf_search.best_estimator_,
        "XGBoost (tuned)": xgb_search.best_estimator_,
    }
    best_name, best_pipeline, best_score = None, None, -1
    for name, pipe in candidates.items():
        score = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="f1_macro").mean()
        print(f"{name}: CV F1(macro)={score:.4f}")
        if score > best_score:
            best_name, best_pipeline, best_score = name, pipe, score

    print(f"\nSelected best model: {best_name} (CV Macro F1 = {best_score:.4f})")

    # --- Final fit on full training set, evaluate on held-out test set ---
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    report_txt = classification_report(y_test, y_pred, target_names=class_names)

    print(f"\nHeld-out test accuracy: {test_acc:.4f}")
    print(f"Held-out test macro F1: {test_f1:.4f}")
    print(report_txt)

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        f"Confusion Matrix — {best_name} (Test Set)",
        os.path.join(ASSET_DIR, "confusion_matrix_best_model.png"),
    )

    # --- Ablation: performance WITHOUT prior grades G1/G2 ---
    print("\n=== Ablation: without G1/G2 (harder, more useful task) ===")
    X_noprior, y_noprior, _ = prepare_dataset(DATA_PATH, drop_prior_grades=True)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(
        X_noprior, y_noprior, test_size=0.2, stratify=y_noprior, random_state=RANDOM_STATE
    )
    prep_noprior = build_preprocessor(drop_prior_grades=True)
    pipe_noprior = Pipeline([("prep", prep_noprior), ("clf", RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE
    ))])
    pipe_noprior.fit(Xtr2, ytr2)
    acc_noprior = accuracy_score(yte2, pipe_noprior.predict(Xte2))
    f1_noprior = f1_score(yte2, pipe_noprior.predict(Xte2), average="macro")
    print(f"Without G1/G2 -> Test Accuracy: {acc_noprior:.4f}, Macro F1: {f1_noprior:.4f}")

    # --- SHAP explainability on the best model ---
    print("\nGenerating SHAP summary plot...")
    feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
    try:
        plot_shap_summary(
            best_pipeline, X_train, feature_names,
            os.path.join(ASSET_DIR, "shap_summary.png"),
        )
        print("SHAP plot saved.")
    except Exception as e:
        print("Could not generate SHAP plot:", e)

    # --- Persist artifacts ---
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(target_encoder, os.path.join(MODEL_DIR, "target_encoder.pkl"))

    results = {
        "baseline_cv_comparison": rows,
        "best_model": best_name,
        "best_model_cv_f1_macro": round(best_score, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4),
        "rf_best_params": rf_search.best_params_,
        "xgb_best_params": xgb_search.best_params_,
        "ablation_no_prior_grades": {
            "test_accuracy": round(acc_noprior, 4),
            "test_f1_macro": round(f1_noprior, 4),
        },
        "classification_report": report_txt,
        "class_names": class_names,
        "feature_columns": list(X.columns),
    }
    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {MODEL_DIR}/best_model.pkl, target_encoder.pkl, results.json")
    print(f"Saved plots to: {ASSET_DIR}/")


if __name__ == "__main__":
    main()
