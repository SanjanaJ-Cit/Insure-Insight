#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

echo "=========================================================="
echo " Starting MediCost AI - Health Insurance Prediction Server"
echo " Currency: Indian Rupees (INR - ₹)"
echo " Model: Random Forest Regressor (R² ~ 95.5%)"
echo " Explainability: SHAP TreeExplainer"
echo " Web UI: http://127.0.0.1:8000"
echo "=========================================================="

python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
