import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# Basic cleaning
df_clean = df.drop(columns=["applicant_id"]).copy()

# Feature engineering
df_clean["was_admitted"] = df_clean["Year_last_admitted"].notna().astype(int)
df_clean["Year_last_admitted"] = df_clean["Year_last_admitted"].fillna(0).astype(int)

median_bmi = float(df_clean["bmi"].median())
df_clean["bmi"] = df_clean["bmi"].fillna(median_bmi)

categorical_cols = [
    "Occupation",
    "cholesterol_level",
    "Gender",
    "smoking_status",
    "Location",
    "covered_by_any_other_company",
    "Alcohol",
    "exercise"
]

numeric_cols = [
    "years_of_insurance_with_us",
    "regular_checkup_lasy_year",
    "adventure_sports",
    "visited_doctor_last_1_year",
    "daily_avg_steps",
    "age",
    "heart_decs_history",
    "other_major_decs_history",
    "avg_glucose_level",
    "bmi",
    "Year_last_admitted",
    "weight",
    "weight_change_in_last_one_year",
    "fat_percentage",
    "was_admitted"
]

# Record category choices and numeric ranges for frontend UI
meta_categories = {}
for c in categorical_cols:
    meta_categories[c] = sorted(df[c].dropna().unique().tolist())

meta_numerics = {}
for n in numeric_cols:
    meta_numerics[n] = {
        "min": float(df_clean[n].min()),
        "max": float(df_clean[n].max()),
        "median": float(df_clean[n].median()),
        "mean": float(df_clean[n].mean())
    }

# Target and features
X_raw = df_clean.drop(columns=["insurance_cost"])
y = df_clean["insurance_cost"].values

# One-hot encoding
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
feature_names = list(X_encoded.columns)

print(f"Dataset shape: {X_encoded.shape}, Features count: {len(feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.15, random_state=42
)

print("Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluation
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

train_r2 = float(r2_score(y_train, y_pred_train))
test_r2 = float(r2_score(y_test, y_pred_test))
test_mae = float(mean_absolute_error(y_test, y_pred_test))
test_rmse = float(root_mean_squared_error(y_test, y_pred_test))

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"Test MAE: ₹{test_mae:,.2f}")
print(f"Test RMSE: ₹{test_rmse:,.2f}")

# SHAP Explainer
print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf)
base_val = explainer.expected_value
if hasattr(base_val, "__iter__"):
    base_val = float(base_val[0])
else:
    base_val = float(base_val)

print(f"SHAP Base Expected Value (Average Premium): ₹{base_val:,.2f}")

# Save artifacts
model_path = os.path.join(ARTIFACTS_DIR, "random_forest_model.joblib")
joblib.dump(rf, model_path, compress=3)
print(f"Model saved to {model_path}")

metadata = {
    "feature_names": feature_names,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "meta_categories": meta_categories,
    "meta_numerics": meta_numerics,
    "median_bmi": median_bmi,
    "base_expected_value": base_val,
    "metrics": {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "total_samples": len(df)
    }
}

meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to {meta_path}")

print("Model training completed successfully!")
