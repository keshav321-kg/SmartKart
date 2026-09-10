# 🛒 SmartKart — Customer Churn Prediction

A complete, beginner-friendly, full-code machine learning pipeline that predicts which SmartKart customers are likely to churn, using Logistic Regression.

**Course:** Introduction to AI & ML | BBA AI/ML | Chitkara Business School
**CLO:** CLO02 — Apply data preprocessing, feature selection & ML models to business scenarios; evaluate performance using appropriate metrics

## Overview

SmartKart is a retail business trying to reduce customer churn. This project takes a deliberately messy 100-row customer dataset (duplicates, missing values, invalid entries, outliers) through a full 15-step ML pipeline and produces a business-ready, ranked churn-risk report.

## Dataset

`SmartKart_dirty_100_rows.csv` — 100 customer records with the following columns:

| Column | Description |
|---|---|
| `Customer_ID` | Unique customer identifier |
| `Age` | Customer age (contains dirty entries: whitespace, text like "thirty", invalid values) |
| `Monthly_Spend` | Average monthly spend (contains outliers, negative values) |
| `Complaints` | Number of complaints filed (contains outliers) |
| `Churn` | Target variable — 1 = churned, 0 = retained |

## Pipeline Steps

1. **Data Collection** — Load the raw CSV
2. **Data Understanding** — Inspect shape, types, missing values, duplicates
3. **Data Cleaning** — Strip whitespace, fix types, remove duplicates/invalid values, impute missing values with the median
4. **Outlier Treatment** — Cap extreme values using the IQR method
5. **Feature Selection** — Keep `Age`, `Monthly_Spend`, `Complaints`; drop `Customer_ID`
6. **Target Definition** — `Churn` (0/1)
7. **Target Encoding** — Verify `Churn` is already numeric
8. **Train-Test Split** — 80/20, stratified
9. **Feature Standardisation** — `StandardScaler`, fit on train only
10. **Model Building** — `LogisticRegression`
11. **Model Training** — Fit on scaled training data
12. **Prediction** — Class predictions + churn probabilities on the test set
13. **Model Evaluation** — Confusion matrix, accuracy, precision, recall, F1-score
14. **Model Interpretation** — Coefficient analysis (which factors drive churn, and why)
15. **Final Output** — Ranked, business-ready churn risk report (CSV)

## Key Findings

- **Higher monthly spend lowers churn risk** — top spenders are the most loyal customers.
- **More complaints strongly increases churn risk** — the clearest actionable lever for SmartKart's retention team.
- Model achieves ~89–95% accuracy with near-100% recall on churners, meaning it rarely misses an actual at-risk customer (a few false alarms are an acceptable trade-off).

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

1. Open `SmartKartipynb.ipynb` in Jupyter or Google Colab.
2. Run all cells in order.
3. When prompted in Step 1, upload `SmartKart_dirty_100_rows.csv`.
4. The final cell exports `smartkart_churn_risk_report.csv` — a ranked list of customers by churn probability, ready for a retention team to act on.

## Output

`smartkart_churn_risk_report.csv` containing, for each test-set customer: features, actual churn, predicted churn, and churn probability — sorted with highest-risk customers first.
