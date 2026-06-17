# Stroke Risk Diagnostic & Prediction Platform

A comprehensive Machine Learning system for predicting stroke probability based on clinical and demographic data. This project implements a robust machine learning pipeline, an Explainable AI (XAI) backend, and a modern frontend interface designed for clinical decision support.

## Live Application
The platform is currently deployed and accessible at: 
**[https://stroke-diagnose-prediction-ml-shap.vercel.app](https://stroke-diagnose-prediction-ml-shap.vercel.app)**

---

## System Architecture

The project has been refactored from a monolithic application into a decoupled, modern architecture:

1. **Machine Learning Pipeline (Jupyter Notebook):** Handles rigorous Exploratory Data Analysis (EDA), feature engineering, and model training.
2. **FastAPI Backend:** Serves the serialized machine learning models and computes SHAP (SHapley Additive exPlanations) values dynamically.
3. **React Frontend:** A highly responsive, modern "Light Mode" user interface designed with Neumorphism and minimalist aesthetics to display risk probabilities and feature impacts.

## Key Features

- **Robust Data Pipeline:** Implements strict Data Leakage prevention by performing Train/Test Splits prior to applying SMOTE (Synthetic Minority Over-sampling Technique) using `imblearn.pipeline.Pipeline`.
- **Ensemble Modeling:** Utilizes multiple classifiers (Random Forest, Gradient Boosting, SVM, Logistic Regression, HistGradientBoosting) culminating in a Soft-Voting Ensemble model for high-accuracy predictions.
- **Explainable AI (SHAP):** Integrates SHAP `KernelExplainer` to calculate the exact impact of each physiological feature (e.g., Glucose level, BMI, Age) on the final stroke probability, ensuring high clinical transparency.
- **Enterprise-Grade UI:** A clean, modern, and accessible user interface built with React and custom CSS, featuring soft drop-shadows, responsive grid layouts, and dynamic horizontal bar charts for SHAP explanations.

## Technology Stack

- **Machine Learning & Data Processing:** Python 3, Scikit-Learn, Pandas, Numpy, Imbalanced-Learn, SHAP
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend Framework:** React, Vite, Axios, Lucide-React
- **Deployment:** Render (Backend), Vercel (Frontend)

---

## Local Installation and Setup

If you wish to run this project locally, follow the instructions below.

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Clone the Repository
```bash
git clone https://github.com/hodatisg520/Stroke-Diagnose-Prediction-ML-SHAP.git
cd Stroke-Diagnose-Prediction-ML-SHAP
```

### 2. Setup the Backend
Navigate to the backend directory, install the required Python packages, and start the FastAPI server.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
The backend API will run on `http://127.0.0.1:8000`.

### 3. Setup the Frontend
Open a new terminal window, navigate to the frontend directory, install dependencies, and start the development server.

```bash
cd frontend-react
npm install
npm run dev
```
The frontend will be accessible at `http://localhost:5173`.

### 4. Machine Learning Training (Optional)
To retrain the models or view the data analysis:
- Open `model_training.ipynb` using Jupyter Notebook or VSCode.
- Run all cells to process the `healthcare-dataset-stroke-data.csv` dataset.
- New `.pkl` model files will be automatically exported to the `backend/` directory.

---

## Disclaimer
This application is designed for educational and research purposes only. It is not intended to substitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
