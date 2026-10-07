# MediCost AI - Medical Insurance Cost Prediction & Health Optimizer (₹ INR Edition)

An intelligent, explainable healthcare economics platform that predicts medical insurance costs using a high-accuracy **Random Forest Regressor (R² = 95.5%)** and employs **SHAP (SHapley Additive exPlanations)** to provide transparent, doctor-friendly lifestyle recommendations to reduce insurance costs in **Indian Rupees (₹)**.

---

## 🌟 Key Features

1. **Random Forest Prediction Engine**:
   - Trained on 25,000 policyholder records from the verified dataset (`Data.csv`).
   - Achieves an $R^2$ of **95.5%** with a low Mean Absolute Error (MAE) of **₹2,421**.
   - Handles multi-variable interactions across 23 demographic, clinical, and lifestyle dimensions.

2. **SHAP (XAI) Factor Decomposition**:
   - Uses `shap.TreeExplainer` for exact mathematical attribution of each feature to the predicted price.
   - Categorizes variables into **Actionable Lifestyle Habits** vs. **Fixed Demographic & Medical History**.
   - Clearly separates **Surcharges (+₹)** that push the premium above the population benchmark (₹27,163) from **Discounts (-₹)**.

3. **Personalized Health Optimization Plan**:
   - Automatically diagnoses suboptimal health markers (e.g., elevated weight/BMI, sedentary steps, smoking, high cholesterol, lack of checkups).
   - Computes **counterfactual scenarios** through the Random Forest model to calculate **exact potential annual savings in ₹ INR** for each health change.
   - Highlights total combined savings (e.g., *"Save up to ₹14,600 / year (38% discount) by adopting recommended healthy habits"*).

4. **Interactive "What-If" Health Simulator**:
   - Real-time sliders allowing users to tweak their target weight, daily steps, smoking status, and exercise routine.
   - Live recalculation updates both the simulated premium and net savings in ₹ instantaneously.

5. **Evidence-Based Health Wellness Score (0 - 100)**:
   - Evaluates metabolic and cardiovascular risk factors into an intuitive gauge meter (Excellent, Good, Moderate Risk, Elevated Risk).

6. **Quick-Load Profiles**:
   - **High Risk**: 42yo Sedentary Smoker, Elevated Weight (Mumbai)
   - **Low Risk**: 26yo Fit & Active Young Adult (Bangalore)
   - **Moderate**: 62yo Senior Citizen with Managed Health (Delhi)

---

## 📂 Project Structure

```
medical_insurance_app/
├── Data.csv                           # Extracted dataset (25,000 records)
├── train_model.py                     # Training pipeline for Random Forest & SHAP
├── recommender.py                     # Counterfactual health recommendation & SHAP logic
├── app.py                             # FastAPI backend API server
├── test_app.py                        # Automated integration test suite
├── run.sh                             # One-click startup script
├── model_artifacts/
│   ├── random_forest_model.joblib     # Persisted Random Forest model
│   └── metadata.json                  # Dataset statistics, metrics, and feature metadata
└── static/
    ├── index.html                     # Responsive single-page application
    ├── css/
    │   └── style.css                  # Custom styling & glassmorphism
    └── js/
        └── app.js                     # Dynamic reactive frontend with Chart.js & Lucide
```

---

## 🚀 How to Run the Website

### Start the Server:
```bash
cd /Users/sanjanaj/.gemini/antigravity/scratch/medical_insurance_app
./run.sh
```
Or directly with Python:
```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Run Automated Tests:
```bash
python3 test_app.py
```
