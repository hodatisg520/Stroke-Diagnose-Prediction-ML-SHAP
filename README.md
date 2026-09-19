
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
- **STROKEGUARD prototype health sync:** Simulated Apple Health, Fitbit, and Garmin connections populate heart rate, blood pressure, sleep, steps, activity, weight, and a small historical trend. These cards are mock integrations for demonstration and do not request real device permissions.
- **AI Doctor prototype:** The new `/ai-doctor` endpoint combines the screening result with manual/device context. It calls Gemini when `GEMINI_API_KEY` is configured and automatically falls back to local rule-based guidance when the key is absent or the service is unavailable.

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
# Optional: enable Gemini-backed guidance. Never commit the real key.
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-2.5-flash-lite"
uvicorn main:app --reload
```
The backend API will run on `http://127.0.0.1:8000`.

Without `GEMINI_API_KEY`, the AI Doctor panel still works in local demo mode. The example variable names are also listed in `backend/.env.example`; set them in your shell or hosting provider rather than placing a secret in the frontend.

### 3. Setup the Frontend
Open a new terminal window, navigate to the frontend directory, install dependencies, and start the development server.

```bash
cd frontend-react
npm install
npm run dev
```
The frontend will be accessible at `http://localhost:5173`.

For a deployed frontend, set `VITE_API_URL` to the public URL of the updated FastAPI backend before building. The Gemini key belongs only in the backend environment, never in a `VITE_` variable.

### 4. Machine Learning Training (Optional)
To retrain the models or view the data analysis:
- Open `model_training.ipynb` using Jupyter Notebook or VSCode.
- Run all cells to process the `healthcare-dataset-stroke-data.csv` dataset.
- New `.pkl` model files will be automatically exported to the `backend/` directory.

---

## Disclaimer
This application and its AI Doctor prototype are designed for educational and research purposes only. They are not intended to substitute professional medical advice, diagnosis, or treatment. The tracker connections are simulated. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition; for sudden emergency symptoms, contact local emergency services.
