# CodeAlpha_CreditScoringModel

**CodeAlpha Machine Learning Internship — Task 1**

## 📌 Objective
Predict an individual's creditworthiness (good vs. bad credit risk) using financial history data such as income, debt, payment history, and credit lines.

## 🧠 Approach
- Feature engineering (e.g. debt-to-income ratio, loan-to-income ratio)
- Three classification models trained and compared:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluation using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**
- ROC curve comparison plot

## 📂 Files
- `credit_scoring_model.py` — full pipeline: data loading, feature engineering, training, evaluation, plotting
- `roc_curves.png` — generated after running the script

## ▶️ How to Run
```bash
pip install numpy pandas scikit-learn matplotlib
python credit_scoring_model.py
```

## 📊 Dataset
The script generates a realistic **synthetic financial dataset** (income, debt, late payments, credit history length, etc.) so it runs out-of-the-box with no download required.

To use a real dataset instead (e.g. UCI German Credit Data or a Kaggle credit dataset), download the CSV and call:
```python
df = load_data(csv_path="your_dataset.csv")
```
Your CSV should include a `default` target column (1 = bad credit risk, 0 = good).

## 📈 Results (example run)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.72 | 0.55 | 0.62 | **0.864** |
| Decision Tree | 0.75 | 0.60 | 0.48 | 0.54 | 0.760 |
| Random Forest | 0.78 | 0.72 | 0.42 | 0.53 | 0.841 |

*(Results vary slightly with random seed / synthetic data regeneration.)*
