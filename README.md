# 🧠 NeuroGuard — Stroke Risk AI

An advanced, multi-model clinical AI application for stroke risk assessment. Built with **Streamlit + scikit-learn + XGBoost + SHAP**, featuring a premium dark UI, interactive Plotly charts, and full model explainability.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green?logo=xgboost)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-cyan?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📌 Overview

This project treats stroke prediction as a **probabilistic classification problem**:

> **P(Stroke | Age, BMI, Hypertension, Glucose, Heart Disease, ...)**

Rather than a simple Yes/No output, the app estimates a continuous stroke probability using **7 different ML models** and presents it alongside clinical-grade analysis, feature explanations, and personalized health advice.

---

## 🤖 Models Implemented

| Model | Type | Key Strength |
|---|---|---|
| **Random Forest** | Ensemble (Bagging) | Robust, interpretable, feature importances |
| **Logistic Regression** | Linear | Interpretable baseline with L2 regularization |
| **Gradient Boosting** | Ensemble (Boosting) | Sequential error correction |
| **XGBoost** | Optimized Boosting | State-of-the-art tabular performance |
| **SVM** | Kernel Method | Strong in high-dimensional spaces |
| **KNN** | Instance-based | Non-parametric, distance-weighted |
| **Voting Ensemble** | Meta-learner | Soft voting over all models, reduces variance |

---

## 🧠 Methodology

| Technique | Details |
|---|---|
| **Imbalance Handling** | SMOTE (Synthetic Minority Over-sampling Technique) |
| **Scaling** | `StandardScaler` inside `sklearn.Pipeline` for SVM/LR/KNN |
| **Evaluation** | 5-Fold Stratified Cross-Validation |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, AUC-ROC |
| **Explainability** | SHAP TreeExplainer (per-prediction feature attribution) |
| **Train/Test Split** | 75% / 25%, stratified |

> ⚠️ **Why not just Accuracy?**  
> With ~4.9% positive rate, a naive "always No Stroke" model hits 95% accuracy but is clinically worthless.  
> We prioritize **Recall** (catching true stroke cases) and **AUC-ROC** (discrimination ability).

---

## 📊 Application Features

### Tab 1 — 🔍 Prediction
- **Risk Score** with color-coded gauge (Low / Moderate / High)
- **Donut chart** showing probability split
- **Feature Importance** bar chart for tree-based models
- **SHAP values** — explains *why* the model made that prediction
- **Clinical Advisory Report** — personalized per risk factor (Glucose, BMI, Hypertension, etc.)

### Tab 2 — 📊 Model Analytics
- **Radar chart** comparing all 7 models across 5 metrics
- **Leaderboard table** with color gradient highlighting
- **ROC Curves** for all models on the same chart
- **Confusion Matrices** for every model
- **Cross-Validation F1 bar chart** with error bars

### Tab 3 — 🗂️ Dataset EDA
- Class distribution donut
- Age & Glucose distributions by stroke status
- Stroke rate by work type
- Feature correlation heatmap
- BMI vs Glucose scatter plot

### Tab 4 — 📖 Methodology
- Dataset details, preprocessing steps
- Per-model descriptions with rationale
- Evaluation strategy explanation
- Full tech stack

---

## 🗂️ Project Structure

```
stroke-diagnose/
├── app_stroke.py                         # Main Streamlit application
├── healthcare-dataset-stroke-data.csv    # Dataset (place here)
├── requirements.txt                      # Dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/stroke-diagnose.git
cd stroke-diagnose
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download from Kaggle and place in root:

```
healthcare-dataset-stroke-data.csv
```

> Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### 5. Run the app

```bash
streamlit run app_stroke.py
```

---

## 📦 Dependencies

```
streamlit
pandas
numpy
matplotlib
scikit-learn
imbalanced-learn
xgboost
plotly
shap
```

---

## 📋 Input Features

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Age | Numeric | 0–100 years |
| Hypertension | Binary | Yes / No |
| Heart Disease | Binary | Yes / No |
| Ever Married | Binary | Yes / No |
| Work Type | Categorical | Private / Self-employed / Govt / Children / Never worked |
| Residence Type | Categorical | Urban / Rural |
| Avg Glucose Level | Numeric | mg/dL |
| BMI | Numeric | kg/m² |
| Smoking Status | Categorical | Never / Formerly / Active / Unknown |

---

## ⚠️ Disclaimer

For **educational and portfolio purposes only**. Not a substitute for professional medical advice. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author
Nguyễn Hồng Đăng
Made with ❤️ as a Data Science portfolio project demonstrating end-to-end ML model selection, evaluation, and deployment skills.
