import os
import json
from app import get_metadata, predict_insurance, simulate, serve_index, ApplicantProfile, WhatIfRequest

def test_meta_endpoint():
    data = get_metadata()
    assert "categories" in data
    assert "Location" in data["categories"]
    assert "Mumbai" in data["categories"]["Location"]
    assert "metrics" in data
    assert data["metrics"]["test_r2"] > 0.90
    print(f"✓ /api/meta passed (Test R²: {data['metrics']['test_r2']:.4f})")

def test_predict_endpoint():
    sample = {
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
    profile = ApplicantProfile(**sample)
    data = predict_insurance(profile)
    assert "predicted_cost" in data
    assert data["currency"] == "INR"
    assert data["currency_symbol"] == "₹"
    assert data["predicted_cost"] > 10000
    assert "shap_breakdown" in data
    assert len(data["shap_breakdown"]["all_features"]) > 0
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    print(f"✓ /api/predict passed (Predicted: ₹{data['predicted_cost']:,.0f}, Potential Savings: ₹{data['combined_scenario']['total_potential_savings']:,.0f})")

def test_simulate_endpoint():
    req = WhatIfRequest(
        profile={
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
        },
        adjustments={
            "weight": 68.0,
            "daily_avg_steps": 9000,
            "smoking_status": "never smoked"
        }
    )
    data = simulate(req)
    assert "simulated_cost" in data
    assert "savings_inr" in data
    print(f"✓ /api/simulate passed (Original: ₹{data['original_cost']:,.0f}, Simulated: ₹{data['simulated_cost']:,.0f}, Savings: ₹{data['savings_inr']:,.0f})")

def test_index_page():
    res = serve_index()
    assert os.path.exists(res.path)
    with open(res.path, "r") as f:
        html = f.read()
    assert "MediCost" in html
    assert "SHAP" in html
    print("✓ GET / (HTML index page exists and valid)")

if __name__ == "__main__":
    print("Running integration tests...")
    test_meta_endpoint()
    test_predict_endpoint()
    test_simulate_endpoint()
    test_index_page()
    print("All tests passed successfully!")
