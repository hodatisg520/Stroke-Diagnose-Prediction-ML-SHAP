from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
import shap

app = FastAPI(title="Stroke Prediction API")

# Setup CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "stroke_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
        feature_cols = meta['feature_cols']
        
    with open(os.path.join(BASE_DIR, "stroke_models.pkl"), "rb") as f:
        models = pickle.load(f)
except Exception as e:
    print(f"Models not found: {e}")
    feature_cols = []
    models = {}

# Prepare a naive background dataset for KernelExplainer (median/mode values could be better, but zeros work for marginal dev)
background_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)

class PatientData(BaseModel):
    gender: str
    age: int
    hypertension: bool
    heart_disease: bool
    ever_married: bool
    avg_glucose_level: float
    bmi: float
    work_type: str
    Residence_type: str
    smoking_status: str
    model_name: str = "Voting Ensemble"

@app.get("/")
def read_root():
    return {"message": "Welcome to Stroke Prediction API"}

@app.get("/models")
def get_models():
    return {"models": list(models.keys())}

@app.post("/predict")
def predict(data: PatientData):
    if data.model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
        
    model = models[data.model_name]
    
    inp_df = pd.DataFrame([{
        'gender': 1 if data.gender == "Female" else 0,
        'age': data.age,
        'hypertension': int(data.hypertension),
        'heart_disease': int(data.heart_disease),
        'ever_married': 1 if data.ever_married else 0,
        'avg_glucose_level': data.avg_glucose_level,
        'bmi': data.bmi,
        'work_type': data.work_type,
        'Residence_type': data.Residence_type,
        'smoking_status': data.smoking_status
    }])
    
    inp_dum = pd.get_dummies(inp_df, columns=['work_type', 'Residence_type', 'smoking_status'])
    inp_fin = inp_dum.reindex(columns=feature_cols, fill_value=0)
    
    try:
        prob = model.predict_proba(inp_fin)[0][1]
        
        if prob < 0.30:
            category = "Low Risk"
        elif prob < 0.60:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        # Calculate SHAP Values using KernelExplainer (approximate and fast enough for 1 sample)
        def predict_proba_wrapper(X_val):
            # Ensure it is a DataFrame to avoid feature name mismatch warnings
            df_temp = pd.DataFrame(X_val, columns=feature_cols)
            return model.predict_proba(df_temp)
            
        explainer = shap.KernelExplainer(predict_proba_wrapper, background_data)
        shap_values = explainer.shap_values(inp_fin)
        
        # KernelExplainer returns a list of arrays for multi-class, index 1 is positive class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            # If shape is (1, N, 2)
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

        # Map to feature names and sort by absolute impact
        feature_impact = []
        for i, col in enumerate(feature_cols):
            # Consolidate dummy variable impacts into base features for cleaner UI
            base_col = col
            if '_' in col and col.split('_')[0] in ['work', 'Residence', 'smoking']:
                base_col = col.rsplit('_', 1)[0]
                
            impact = float(sv[i])
            # Only add if impact is significant
            if abs(impact) > 0.001:
                feature_impact.append({"feature": col, "impact": impact})
                
        # Sort by absolute impact descending
        feature_impact.sort(key=lambda x: abs(x['impact']), reverse=True)
            
        return {
            "model_used": data.model_name,
            "stroke_probability": float(prob),
            "risk_category": category,
            "shap_explanation": feature_impact[:10] # Top 10 impactful features
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
