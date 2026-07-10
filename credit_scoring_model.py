"""
CodeAlpha Machine Learning Internship - Task 1
Credit Scoring Model
------------------------------------------------
Predicts an individual's creditworthiness (good/bad credit risk) using
past financial data such as income, debts, and payment history.

Approach: Logistic Regression, Decision Tree, and Random Forest classifiers
are trained and compared using Precision, Recall, F1-Score, and ROC-AUC.

Author: (your name here)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display / in CI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------
# 1. DATA LOADING / GENERATION
# --------------------------------------------------------------------------
def load_data(csv_path: str = None, n_samples: int = 2000) -> pd.DataFrame:
    """
    Loads a real credit dataset if a CSV path is provided (e.g. the UCI
    German Credit Data or a Kaggle credit-scoring dataset with columns such
    as income, debt, payment_history, etc.). If no path is given, a
    realistic synthetic dataset is generated so the pipeline can be run
    and demonstrated end-to-end without needing an external download.

    Expected columns if you bring your own CSV:
        age, income, debt, credit_lines, loan_amount,
        late_payments, employment_years, credit_history_length, default
    where 'default' is the target column (1 = bad credit risk, 0 = good).
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        return df

    # ---- Synthetic but realistic financial data ----
    age = np.random.randint(21, 65, n_samples)
    income = np.random.normal(55000, 20000, n_samples).clip(12000, 200000)
    debt = np.random.normal(15000, 10000, n_samples).clip(0, 150000)
    credit_lines = np.random.randint(0, 15, n_samples)
    loan_amount = np.random.normal(20000, 15000, n_samples).clip(500, 150000)
    late_payments = np.random.poisson(1.2, n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    credit_history_length = np.random.randint(0, 30, n_samples)

    debt_to_income = debt / income

    # Underlying "true" risk score used to generate a realistic target
    risk_score = (
        0.000009 * debt
        - 0.00002 * income
        + 0.35 * late_payments
        + 0.9 * debt_to_income
        - 0.05 * employment_years
        - 0.03 * credit_history_length
        + 0.06 * credit_lines
        + np.random.normal(0, 1.0, n_samples)  # noise
    )
    threshold = np.percentile(risk_score, 70)  # ~30% default rate
    default = (risk_score > threshold).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "debt": debt.round(2),
            "credit_lines": credit_lines,
            "loan_amount": loan_amount.round(2),
            "late_payments": late_payments,
            "employment_years": employment_years,
            "credit_history_length": credit_history_length,
            "debt_to_income": debt_to_income.round(4),
            "default": default,
        }
    )
    return df


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "debt_to_income" not in df.columns:
        df["debt_to_income"] = df["debt"] / df["income"].replace(0, np.nan)
        df["debt_to_income"] = df["debt_to_income"].fillna(0)
    df["loan_to_income"] = df["loan_amount"] / df["income"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_to_income"].fillna(0)
    return df


# --------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc, "y_proba": y_proba}


def plot_roc_curves(results, y_test, out_path="roc_curves.png"):
    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC = {r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Credit Scoring Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nROC curve plot saved to {out_path}")


def main():
    print("Loading data...")
    df = load_data()  # pass csv_path="your_dataset.csv" to use a real dataset
    df = engineer_features(df)

    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            res = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            res = evaluate_model(name, model, X_test, y_test)
        results.append(res)

    plot_roc_curves(results, y_test)

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best['name']} ({best['roc_auc']:.4f})")


if __name__ == "__main__":
    main()
