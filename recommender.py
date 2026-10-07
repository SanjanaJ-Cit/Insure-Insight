import os
import json
import numpy as np
import pandas as pd
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")

# Load artifacts
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "random_forest_model.joblib")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")

print("Loading model and metadata...")
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    meta = json.load(f)

feature_names = meta["feature_names"]
categorical_cols = meta["categorical_cols"]
numeric_cols = meta["numeric_cols"]
median_bmi = meta["median_bmi"]
base_expected_value = meta["base_expected_value"]

# Precompute SHAP TreeExplainer
explainer = shap.TreeExplainer(model)

FEATURE_DISPLAY_NAMES = {
    "weight": "Body Weight (kg)",
    "daily_avg_steps": "Daily Step Count",
    "bmi": "Body Mass Index (BMI)",
    "smoking_status": "Smoking Habit",
    "avg_glucose_level": "Average Blood Glucose (mg/dL)",
    "cholesterol_level": "Cholesterol Level (mg/dL)",
    "exercise": "Physical Exercise Routine",
    "Alcohol": "Alcohol Consumption",
    "regular_checkup_lasy_year": "Annual Preventative Checkups",
    "visited_doctor_last_1_year": "Doctor Visits (Past Year)",
    "weight_change_in_last_one_year": "Weight Change in Past Year (kg)",
    "fat_percentage": "Body Fat Percentage (%)",
    "age": "Age (Years)",
    "Gender": "Gender",
    "Occupation": "Employment Category",
    "Location": "Geographic City / Region",
    "heart_decs_history": "Heart Condition History",
    "other_major_decs_history": "Major Medical Condition History",
    "was_admitted": "Past Hospitalization History",
    "Year_last_admitted": "Year of Last Hospitalization",
    "years_of_insurance_with_us": "Tenure with Insurance Provider (Years)",
    "covered_by_any_other_company": "Existing Additional Coverage",
    "adventure_sports": "High-Risk Adventure Sports"
}

ACTIONABLE_FEATURES = {
    "weight", "daily_avg_steps", "bmi", "smoking_status", "avg_glucose_level",
    "cholesterol_level", "exercise", "Alcohol", "regular_checkup_lasy_year", "fat_percentage"
}

def preprocess_input(user_dict: dict) -> pd.DataFrame:
    """Transform user input dictionary into one-hot encoded DataFrame matching model feature columns."""
    df_raw = pd.DataFrame([user_dict.copy()])
    
    # Process admission info
    if "Year_last_admitted" in df_raw.columns and pd.notna(df_raw["Year_last_admitted"].values[0]) and float(df_raw["Year_last_admitted"].values[0]) > 0:
        df_raw["was_admitted"] = 1
        df_raw["Year_last_admitted"] = int(df_raw["Year_last_admitted"].values[0])
    else:
        df_raw["was_admitted"] = 0
        df_raw["Year_last_admitted"] = 0
        
    # BMI imputation if not provided
    if "bmi" not in df_raw.columns or pd.isna(df_raw["bmi"].values[0]) or float(df_raw["bmi"].values[0]) <= 0:
        # If height is given or estimate from weight
        df_raw["bmi"] = median_bmi
        
    # Ensure all numeric cols have correct types
    for num_col in numeric_cols:
        if num_col in df_raw.columns:
            df_raw[num_col] = pd.to_numeric(df_raw[num_col], errors="coerce").fillna(meta["meta_numerics"].get(num_col, {}).get("median", 0))
            
    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df_raw, columns=categorical_cols, drop_first=False)
    
    # Align columns with model
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_aligned = df_encoded[feature_names].astype(float)
    return df_aligned

def calculate_health_score(profile: dict) -> dict:
    """Calculate an evidence-based wellness & risk score from 0 to 100."""
    score = 100
    deductions = []
    
    # Weight & BMI
    bmi = float(profile.get("bmi", 25))
    weight = float(profile.get("weight", 70))
    if bmi > 30 or weight > 82:
        score -= 20
        deductions.append("High BMI / Elevated Body Weight (-20 pts)")
    elif bmi > 25 or weight > 74:
        score -= 10
        deductions.append("Overweight range (-10 pts)")
        
    # Steps
    steps = int(profile.get("daily_avg_steps", 5000))
    if steps < 4000:
        score -= 15
        deductions.append("Sedentary lifestyle (<4,000 daily steps) (-15 pts)")
    elif steps < 7000:
        score -= 7
        deductions.append("Moderate physical inactivity (<7,000 steps) (-7 pts)")
        
    # Smoking
    smoking = str(profile.get("smoking_status", "never smoked"))
    if smoking == "smokes":
        score -= 22
        deductions.append("Active smoker (-22 pts)")
    elif smoking == "formerly smoked":
        score -= 8
        deductions.append("Past smoking history (-8 pts)")
        
    # Alcohol
    alcohol = str(profile.get("Alcohol", "No"))
    if alcohol == "Daily":
        score -= 14
        deductions.append("Daily alcohol intake (-14 pts)")
    elif alcohol == "Rare":
        score -= 3
        deductions.append("Occasional alcohol consumption (-3 pts)")
        
    # Exercise
    exercise = str(profile.get("exercise", "No"))
    if exercise == "No":
        score -= 10
        deductions.append("Zero routine exercise (-10 pts)")
    elif exercise == "Moderate":
        score += 2 # slight bonus
        
    # Glucose
    glucose = float(profile.get("avg_glucose_level", 100))
    if glucose > 180:
        score -= 15
        deductions.append("Significantly elevated glucose (>180 mg/dL) (-15 pts)")
    elif glucose > 130:
        score -= 8
        deductions.append("Borderline prediabetic glucose (>130 mg/dL) (-8 pts)")
        
    # Cholesterol
    chol = str(profile.get("cholesterol_level", "125 to 150"))
    if "225 to 250" in chol or "200 to 225" in chol:
        score -= 10
        deductions.append("High blood cholesterol (-10 pts)")
        
    # Medical history
    if int(profile.get("heart_decs_history", 0)) == 1:
        score -= 15
        deductions.append("Heart condition history (-15 pts)")
    if int(profile.get("other_major_decs_history", 0)) == 1:
        score -= 12
        deductions.append("Major medical condition history (-12 pts)")
        
    score = max(10, min(99, score))
    
    if score >= 80:
        grade = "Excellent"
        color = "#10B981" # Green
        description = "You have an outstanding health and lifestyle profile with low actuarial insurance risk."
    elif score >= 65:
        grade = "Good"
        color = "#3B82F6" # Blue
        description = "Your health profile is solid, with a few moderate lifestyle optimization opportunities."
    elif score >= 45:
        grade = "Moderate Risk"
        color = "#F59E0B" # Amber
        description = "Several lifestyle and clinical markers are elevating your health insurance premium."
    else:
        grade = "Elevated Risk"
        color = "#EF4444" # Red
        description = "Significant metabolic and lifestyle factors are driving higher health insurance pricing."
        
    return {
        "score": score,
        "grade": grade,
        "color": color,
        "description": description,
        "deductions": deductions
    }

def get_prediction_and_explanation(user_dict: dict) -> dict:
    """Predict premium in INR, compute SHAP feature contributions, and provide actionable suggestions."""
    X_sample = preprocess_input(user_dict)
    
    # 1. Prediction
    predicted_cost = float(model.predict(X_sample)[0])
    # Round to realistic rupee figure
    predicted_cost = round(predicted_cost, 0)
    monthly_cost = round(predicted_cost / 12, 0)
    
    # 2. SHAP Explanation
    shap_explanation = explainer(X_sample)
    raw_shap_values = shap_explanation.values[0]
    
    # Aggregate one-hot SHAP values back to original feature names
    feature_impacts = {}
    
    for feat_col, val in zip(feature_names, raw_shap_values):
        # Check if one-hot column
        matched_cat = None
        for cat in categorical_cols:
            if feat_col.startswith(cat + "_"):
                matched_cat = cat
                break
        
        feature_key = matched_cat if matched_cat else feat_col
        feature_impacts[feature_key] = feature_impacts.get(feature_key, 0.0) + float(val)
        
    # Format SHAP features list
    explained_features = []
    for k, v in feature_impacts.items():
        if abs(v) < 1.0: # Filter out negligible noise
            continue
            
        display_name = FEATURE_DISPLAY_NAMES.get(k, k.replace("_", " ").title())
        is_actionable = k in ACTIONABLE_FEATURES
        user_val = user_dict.get(k, "N/A")
        
        # User-friendly explanation string
        if v > 0:
            impact_type = "Increases Cost"
            summary = f"Adds ₹{v:,.0f} to your premium due to higher risk"
        else:
            impact_type = "Reduces Cost"
            summary = f"Saves ₹{abs(v):,.0f} on your premium due to healthy factor"
            
        explained_features.append({
            "key": k,
            "name": display_name,
            "value": str(user_val),
            "shap_value": round(v, 2),
            "impact_inr": round(v, 0),
            "abs_impact": abs(round(v, 0)),
            "impact_type": impact_type,
            "is_actionable": is_actionable,
            "summary": summary
        })
        
    # Sort by absolute SHAP impact
    explained_features.sort(key=lambda x: x["abs_impact"], reverse=True)
    
    cost_pushers = [f for f in explained_features if f["shap_value"] > 0]
    cost_savers = [f for f in explained_features if f["shap_value"] < 0]
    
    # 3. Health Score
    health_score_data = calculate_health_score(user_dict)
    
    # 4. Actionable Health & Cost-Reduction Recommendations Engine
    recommendations = []
    
    current_weight = float(user_dict.get("weight", 70))
    current_steps = int(user_dict.get("daily_avg_steps", 5000))
    current_smoking = str(user_dict.get("smoking_status", "never smoked"))
    current_alcohol = str(user_dict.get("Alcohol", "No"))
    current_exercise = str(user_dict.get("exercise", "No"))
    current_cholesterol = str(user_dict.get("cholesterol_level", "125 to 150"))
    current_glucose = float(user_dict.get("avg_glucose_level", 100))
    current_checkup = int(user_dict.get("regular_checkup_lasy_year", 0))
    
    # Helper for counterfactual evaluation
    def evaluate_counterfactual(modifications: dict) -> float:
        modified_dict = user_dict.copy()
        modified_dict.update(modifications)
        X_mod = preprocess_input(modified_dict)
        return float(model.predict(X_mod)[0])
        
    # A. Weight Reduction recommendation
    if current_weight > 62:
        # Target weight: healthy 10-15% reduction or down to ~60-65kg
        target_weight = max(55.0, round(current_weight * 0.88, 1))
        weight_savings = predicted_cost - evaluate_counterfactual({
            "weight": target_weight,
            "bmi": max(21.0, float(user_dict.get("bmi", 28)) * 0.88),
            "fat_percentage": max(15.0, float(user_dict.get("fat_percentage", 28)) * 0.85)
        })
        if weight_savings > 100:
            recommendations.append({
                "id": "weight_optimization",
                "category": "Weight & Metabolism",
                "title": f"Target Weight Optimization (from {current_weight} kg to {target_weight} kg)",
                "action": f"Gradual calorie deficit & portion control to reach {target_weight} kg",
                "savings_inr": round(weight_savings, 0),
                "savings_pct": round((weight_savings / predicted_cost) * 100, 1),
                "badge": "Highest Impact",
                "icon": "scale",
                "advice": "In health insurance actuarial models, body weight has the largest influence on cardiovascular and metabolic risk. Losing 5-10% body weight drops you into lower risk underwriting brackets."
            })
            
    # B. Physical Activity / Daily Steps
    if current_steps < 8000:
        target_steps = 8500
        steps_savings = predicted_cost - evaluate_counterfactual({"daily_avg_steps": target_steps, "exercise": "Moderate" if current_exercise == "No" else current_exercise})
        if steps_savings > 50:
            recommendations.append({
                "id": "steps_increase",
                "category": "Physical Activity",
                "title": f"Increase Daily Steps (from {current_steps:,} to {target_steps:,} steps/day)",
                "action": "Aim for 30 minutes of brisk walking each morning or evening",
                "savings_inr": round(steps_savings, 0),
                "savings_pct": round((steps_savings / max(1, predicted_cost)) * 100, 1),
                "badge": "High Health Benefit",
                "icon": "footprints",
                "advice": "Walking 8,500+ steps per day lowers arterial stiffness, improves insulin sensitivity, and provides insurer-verified wellness credits."
            })
            
    # C. Smoking Cessation
    if current_smoking in ["smokes", "formerly smoked"]:
        smoke_savings = predicted_cost - evaluate_counterfactual({"smoking_status": "never smoked"})
        # Insurers consider non-smokers substantially lower risk
        effective_smoke_savings = max(1500.0, smoke_savings) if current_smoking == "smokes" else max(600.0, smoke_savings)
        recommendations.append({
            "id": "smoking_cessation",
            "category": "Cardiopulmonary",
            "title": "Smoking Cessation Program",
            "action": "Complete tobacco cessation or nicotine replacement therapy",
            "savings_inr": round(effective_smoke_savings, 0),
            "savings_pct": round((effective_smoke_savings / predicted_cost) * 100, 1),
            "badge": "Critical Wellness",
            "icon": "ban",
            "advice": "Quitting smoking halves cardiac risk within 1 year and drastically reduces critical illness claim probabilities."
        })
        
    # D. Exercise Routine
    if current_exercise == "No":
        exercise_savings = predicted_cost - evaluate_counterfactual({"exercise": "Moderate", "daily_avg_steps": max(6500, current_steps)})
        if exercise_savings > 50:
            recommendations.append({
                "id": "exercise_routine",
                "category": "Fitness & Endurance",
                "title": "Start Moderate Exercise Routine (3-4 days/week)",
                "action": "150 minutes per week of aerobic cardio or strength workouts",
                "savings_inr": round(exercise_savings, 0),
                "savings_pct": round((exercise_savings / max(1, predicted_cost)) * 100, 1),
                "badge": "Active Living",
                "icon": "dumbbell",
                "advice": "Regular aerobic exercise stabilizes resting heart rate, regulates blood pressure, and reduces sedentary illness underwriting surcharges."
            })
            
    # E. Alcohol Moderation
    if current_alcohol == "Daily":
        alcohol_savings = predicted_cost - evaluate_counterfactual({"Alcohol": "No"})
        effective_alc_savings = max(800.0, alcohol_savings)
        recommendations.append({
            "id": "alcohol_moderation",
            "category": "Liver & Metabolic",
            "title": "Moderate Alcohol Intake (Switch from Daily to Rare/None)",
            "action": "Limit drinks to social occasions or eliminate completely",
            "savings_inr": round(effective_alc_savings, 0),
            "savings_pct": round((effective_alc_savings / predicted_cost) * 100, 1),
            "badge": "Metabolic Care",
            "icon": "wine-off",
            "advice": "Daily alcohol intake stresses hepatic enzymes and increases hypertension risks. Curtailing intake helps avoid clinical copay penalties."
        })
        
    # F. Cholesterol Management
    if current_cholesterol in ["200 to 225", "225 to 250", "175 to 200"]:
        chol_savings = predicted_cost - evaluate_counterfactual({"cholesterol_level": "125 to 150"})
        if chol_savings > 50:
            recommendations.append({
                "id": "cholesterol_reduction",
                "category": "Cardiovascular Lipid Health",
                "title": f"Lower Cholesterol to Healthy Range (125 - 150 mg/dL)",
                "action": "Adopt a Mediterranean diet rich in oats, nuts, legumes, and omega-3s",
                "savings_inr": round(chol_savings, 0),
                "savings_pct": round((chol_savings / max(1, predicted_cost)) * 100, 1),
                "badge": "Heart Protective",
                "icon": "heart-pulse",
                "advice": "High LDL cholesterol causes arterial plaque formation. Keeping total cholesterol under 175 mg/dL drastically reduces stroke & CAD risk."
            })
            
    # G. Blood Glucose Normalization
    if current_glucose > 130:
        glucose_savings = predicted_cost - evaluate_counterfactual({"avg_glucose_level": 95.0})
        if glucose_savings > 50:
            recommendations.append({
                "id": "glucose_management",
                "category": "Glycemic & Diabetes Prevention",
                "title": f"Normalize Blood Glucose (from {current_glucose} to <100 mg/dL)",
                "action": "Reduce refined sugars, increase dietary fiber, and monitor HbA1c",
                "savings_inr": round(glucose_savings, 0),
                "savings_pct": round((glucose_savings / max(1, predicted_cost)) * 100, 1),
                "badge": "Endocrine Control",
                "icon": "activity",
                "advice": "Fasting blood glucose above 126 mg/dL indicates diabetes risk, which attracts insurance loading. Normalizing glucose keeps premiums at baseline rates."
            })
            
    # H. Annual Preventative Checkups
    if current_checkup == 0:
        checkup_savings = predicted_cost - evaluate_counterfactual({"regular_checkup_lasy_year": 2})
        if checkup_savings > 50:
            recommendations.append({
                "id": "regular_screenings",
                "category": "Preventative Medicine",
                "title": "Undergo Regular Annual Health Screenings",
                "action": "Schedule comprehensive annual preventive body checkup",
                "savings_inr": round(checkup_savings, 0),
                "savings_pct": round((checkup_savings / max(1, predicted_cost)) * 100, 1),
                "badge": "Incentive Credit",
                "icon": "stethoscope",
                "advice": "Insurance providers reward policyholders who conduct annual checkups because early condition detection drastically lowers expensive hospitalization."
            })
            
    # Calculate Combined Optimal Lifestyle Profile
    combined_ideal = user_dict.copy()
    if current_weight > 62:
        combined_ideal["weight"] = max(55.0, round(current_weight * 0.88, 1))
        combined_ideal["bmi"] = max(21.0, float(user_dict.get("bmi", 28)) * 0.88)
        combined_ideal["fat_percentage"] = max(15.0, float(user_dict.get("fat_percentage", 28)) * 0.85)
    if current_steps < 8000:
        combined_ideal["daily_avg_steps"] = 8500
    if current_smoking in ["smokes", "formerly smoked"]:
        combined_ideal["smoking_status"] = "never smoked"
    if current_alcohol == "Daily":
        combined_ideal["Alcohol"] = "No"
    combined_ideal["exercise"] = "Moderate"
    combined_ideal["cholesterol_level"] = "125 to 150"
    if current_glucose > 120:
        combined_ideal["avg_glucose_level"] = 95.0
    combined_ideal["regular_checkup_lasy_year"] = max(1, current_checkup)
    
    optimal_cost = round(float(model.predict(preprocess_input(combined_ideal))[0]), 0)
    total_potential_savings = max(0.0, round(predicted_cost - optimal_cost, 0))
    total_savings_pct = round((total_potential_savings / max(1, predicted_cost)) * 100, 1) if predicted_cost > 0 else 0
    
    # Sort recommendations by savings INR descending
    recommendations.sort(key=lambda x: x["savings_inr"], reverse=True)
    
    return {
        "predicted_cost": predicted_cost,
        "monthly_cost": monthly_cost,
        "base_expected_value": round(base_expected_value, 0),
        "cost_difference_from_average": round(predicted_cost - base_expected_value, 0),
        "currency": "INR",
        "currency_symbol": "₹",
        "health_score": health_score_data,
        "shap_breakdown": {
            "all_features": explained_features,
            "cost_pushers": cost_pushers[:6],
            "cost_savers": cost_savers[:6],
            "base_value": round(base_expected_value, 0)
        },
        "recommendations": recommendations,
        "combined_scenario": {
            "optimized_cost": optimal_cost,
            "total_potential_savings": total_potential_savings,
            "total_savings_pct": total_savings_pct
        }
    }

def simulate_what_if(user_dict: dict, adjustments: dict) -> dict:
    """Live What-If analysis: adjust specific health levers and return instantaneous delta in INR."""
    current_pred = round(float(model.predict(preprocess_input(user_dict))[0]), 0)
    
    simulated_profile = user_dict.copy()
    simulated_profile.update(adjustments)
    
    new_pred = round(float(model.predict(preprocess_input(simulated_profile))[0]), 0)
    delta_inr = round(current_pred - new_pred, 0)
    delta_pct = round((delta_inr / max(1, current_pred)) * 100, 1)
    
    return {
        "original_cost": current_pred,
        "simulated_cost": new_pred,
        "savings_inr": delta_inr,
        "savings_pct": delta_pct,
        "currency_symbol": "₹"
    }

if __name__ == "__main__":
    # Test sample
    test_user = {
        "years_of_insurance_with_us": 2,
        "regular_checkup_lasy_year": 0,
        "adventure_sports": 0,
        "Occupation": "Salried",
        "visited_doctor_last_1_year": 3,
        "cholesterol_level": "200 to 225",
        "daily_avg_steps": 4200,
        "age": 42,
        "heart_decs_history": 0,
        "other_major_decs_history": 0,
        "Gender": "Male",
        "avg_glucose_level": 165,
        "bmi": 32.5,
        "smoking_status": "smokes",
        "Year_last_admitted": 0,
        "Location": "Mumbai",
        "weight": 79,
        "covered_by_any_other_company": "N",
        "Alcohol": "Daily",
        "exercise": "No",
        "weight_change_in_last_one_year": 2,
        "fat_percentage": 33
    }
    
    res = get_prediction_and_explanation(test_user)
    print("\n--- TEST PREDICTION & EXPLANATION ---")
    print(f"Predicted Cost: ₹{res['predicted_cost']:,.0f} / yr (₹{res['monthly_cost']:,.0f} / mo)")
    print(f"Health Score: {res['health_score']['score']} ({res['health_score']['grade']})")
    print(f"Total Potential Savings: ₹{res['combined_scenario']['total_potential_savings']:,.0f} ({res['combined_scenario']['total_savings_pct']}%)")
    print("\nTop Cost Pushers:")
    for cp in res["shap_breakdown"]["cost_pushers"][:3]:
        print(f" - {cp['name']} ({cp['value']}): +₹{cp['impact_inr']:,.0f}")
    print("\nRecommendations:")
    for r in res["recommendations"][:4]:
        print(f" * {r['title']}: Save ₹{r['savings_inr']:,.0f} ({r['savings_pct']}%)")
