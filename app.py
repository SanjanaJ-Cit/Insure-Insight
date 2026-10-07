import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from recommender import (
    get_prediction_and_explanation,
    simulate_what_if,
    meta,
    FEATURE_DISPLAY_NAMES,
    ACTIONABLE_FEATURES
)

app = FastAPI(
    title="MediCost AI - Medical Insurance Cost Prediction & Health Optimizer",
    description="Random Forest ML model with SHAP Explainable AI for Health Insurance Pricing & Savings in Indian Rupees (₹)",
    version="2.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Pydantic schemas
class ApplicantProfile(BaseModel):
    age: int = Field(default=35, ge=16, le=90, description="Age in years")
    Gender: str = Field(default="Male", description="Gender (Male / Female)")
    Occupation: str = Field(default="Salried", description="Occupation")
    Location: str = Field(default="Bangalore", description="City location")
    weight: float = Field(default=72.0, ge=40.0, le=140.0, description="Weight in kg")
    bmi: Optional[float] = Field(default=25.5, ge=10.0, le=60.0, description="Body Mass Index")
    fat_percentage: float = Field(default=26.0, ge=5.0, le=55.0, description="Body fat percentage")
    daily_avg_steps: int = Field(default=5500, ge=1000, le=25000, description="Daily steps")
    exercise: str = Field(default="Moderate", description="Exercise frequency")
    smoking_status: str = Field(default="never smoked", description="Smoking habit")
    Alcohol: str = Field(default="Rare", description="Alcohol intake")
    cholesterol_level: str = Field(default="150 to 175", description="Cholesterol range")
    avg_glucose_level: float = Field(default=110.0, ge=50.0, le=350.0, description="Avg glucose mg/dL")
    regular_checkup_lasy_year: int = Field(default=1, ge=0, le=10, description="Preventative checkups")
    visited_doctor_last_1_year: int = Field(default=2, ge=0, le=20, description="Doctor visits")
    weight_change_in_last_one_year: float = Field(default=1.0, ge=0.0, le=20.0, description="Weight delta")
    years_of_insurance_with_us: int = Field(default=2, ge=0, le=15, description="Policy tenure")
    covered_by_any_other_company: str = Field(default="N", description="Covered by another insurer")
    adventure_sports: int = Field(default=0, ge=0, le=1, description="Extreme sports participant")
    heart_decs_history: int = Field(default=0, ge=0, le=1, description="Heart condition history")
    other_major_decs_history: int = Field(default=0, ge=0, le=1, description="Major disease history")
    Year_last_admitted: Optional[float] = Field(default=0.0, description="Year last admitted (0 if never)")

class WhatIfRequest(BaseModel):
    profile: Dict[str, Any]
    adjustments: Dict[str, Any]

@app.get("/api/meta")
def get_metadata():
    """Return model parameters, ranges, categorical options, and preset sample profiles."""
    presets = [
        {
            "id": "elevated_risk",
            "name": "Sedentary Professional (High Cost)",
            "subtitle": "42 yrs, 79 kg, Smoker, Sedentary, Mumbai",
            "data": {
                "age": 42,
                "Gender": "Male",
                "Occupation": "Salried",
                "Location": "Mumbai",
                "weight": 79.0,
                "bmi": 32.5,
                "fat_percentage": 33.0,
                "daily_avg_steps": 3800,
                "exercise": "No",
                "smoking_status": "smokes",
                "Alcohol": "Daily",
                "cholesterol_level": "200 to 225",
                "avg_glucose_level": 165.0,
                "regular_checkup_lasy_year": 0,
                "visited_doctor_last_1_year": 4,
                "weight_change_in_last_one_year": 3.0,
                "years_of_insurance_with_us": 1,
                "covered_by_any_other_company": "N",
                "adventure_sports": 0,
                "heart_decs_history": 0,
                "other_major_decs_history": 0,
                "Year_last_admitted": 0
            }
        },
        {
            "id": "healthy_active",
            "name": "Active & Fit Young Adult (Low Cost)",
            "subtitle": "26 yrs, 58 kg, 9,500 Steps, Non-Smoker, Bangalore",
            "data": {
                "age": 26,
                "Gender": "Female",
                "Occupation": "Salried",
                "Location": "Bangalore",
                "weight": 58.0,
                "bmi": 21.5,
                "fat_percentage": 19.0,
                "daily_avg_steps": 9500,
                "exercise": "Moderate",
                "smoking_status": "never smoked",
                "Alcohol": "No",
                "cholesterol_level": "125 to 150",
                "avg_glucose_level": 88.0,
                "regular_checkup_lasy_year": 2,
                "visited_doctor_last_1_year": 1,
                "weight_change_in_last_one_year": 0.0,
                "years_of_insurance_with_us": 3,
                "covered_by_any_other_company": "N",
                "adventure_sports": 0,
                "heart_decs_history": 0,
                "other_major_decs_history": 0,
                "Year_last_admitted": 0
            }
        },
        {
            "id": "senior_moderate",
            "name": "Senior Citizen with Managed Health (Standard Cost)",
            "subtitle": "62 yrs, 68 kg, Moderate Walk, Former Smoker, Delhi",
            "data": {
                "age": 62,
                "Gender": "Male",
                "Occupation": "Business",
                "Location": "Delhi",
                "weight": 68.0,
                "bmi": 26.2,
                "fat_percentage": 27.0,
                "daily_avg_steps": 6200,
                "exercise": "Moderate",
                "smoking_status": "formerly smoked",
                "Alcohol": "Rare",
                "cholesterol_level": "175 to 200",
                "avg_glucose_level": 128.0,
                "regular_checkup_lasy_year": 2,
                "visited_doctor_last_1_year": 3,
                "weight_change_in_last_one_year": 1.0,
                "years_of_insurance_with_us": 6,
                "covered_by_any_other_company": "Y",
                "adventure_sports": 0,
                "heart_decs_history": 0,
                "other_major_decs_history": 0,
                "Year_last_admitted": 2012
            }
        }
    ]
    
    return {
        "categories": meta["meta_categories"],
        "numerics": meta["meta_numerics"],
        "metrics": meta["metrics"],
        "base_expected_value": meta["base_expected_value"],
        "feature_labels": FEATURE_DISPLAY_NAMES,
        "actionable_features": list(ACTIONABLE_FEATURES),
        "presets": presets
    }

@app.post("/api/predict")
def predict_insurance(profile: ApplicantProfile):
    """Predict premium in INR, compute SHAP explainability, and generate actionable health recommendations."""
    try:
        user_data = profile.model_dump()
        result = get_prediction_and_explanation(user_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def simulate(req: WhatIfRequest):
    """Compute real-time counterfactual premium and cost savings in INR."""
    try:
        res = simulate_what_if(req.profile, req.adjustments)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
