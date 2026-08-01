import os
import pickle
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Xgboost_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None

# Custom CSS + HTML Template for a clean interactive interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Status Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f0f2f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding-bottom: 40px;
        }
        .main-card {
            background: #ffffff;
            border: none;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            margin-top: 40px;
            padding: 30px;
        }
        .header-title {
            color: #1a2b4c;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .section-badge {
            background-color: #eef2ff;
            color: #4f46e5;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 15px;
            font-size: 0.85rem;
        }
        .form-label {
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9rem;
        }
        .form-control, .form-select {
            border-radius: 8px;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
        }
        .form-control:focus, .form-select:focus {
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        .btn-predict {
            background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            font-weight: 600;
            border-radius: 10px;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3);
            color: white;
        }
        .result-box {
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            text-align: center;
        }
        .result-approved {
            background-color: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }
        .result-rejected {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-card">
                    <div class="text-center mb-4">
                        <h2 class="header-title">🏦 Loan Approval Prediction</h2>
                        <p class="text-muted">Enter applicant financial and credit parameters to predict eligibility.</p>
                    </div>

                    {% if error_msg %}
                    <div class="alert alert-danger" role="alert">
                        <strong>Error:</strong> {{ error_msg }}
                    </div>
                    {% endif %}

                    <form method="POST" action="/">
                        <div class="row g-4">
                            <!-- Column 1: Personal Info -->
                            <div class="col-md-4">
                                <span class="section-badge">1. Personal Info</span>
                                <div class="mb-3">
                                    <label class="form-label">Number of Dependents</label>
                                    <input type="number" name="no_of_dependents" class="form-control" value="{{ form.get('no_of_dependents', 2) }}" min="0" max="20" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Education</label>
                                    <select name="education" class="form-select">
                                        <option value="Graduate" {% if form.get('education') == 'Graduate' or not form %}selected{% endif %}>Graduate</option>
                                        <option value="Not Graduate" {% if form.get('education') == 'Not Graduate' %}selected{% endif %}>Not Graduate</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Self Employed</label>
                                    <select name="self_employed" class="form-select">
                                        <option value="No" {% if form.get('self_employed') == 'No' or not form %}selected{% endif %}>No</option>
                                        <option value="Yes" {% if form.get('self_employed') == 'Yes' %}selected{% endif %}>Yes</option>
                                    </select>
                                </div>
                            </div>

                            <!-- Column 2: Income & Credit -->
                            <div class="col-md-4">
                                <span class="section-badge">2. Loan & Credit</span>
                                <div class="mb-3">
                                    <label class="form-label">Annual Income (₹/$)</label>
                                    <input type="number" name="income_annum" class="form-control" value="{{ form.get('income_annum', 5000000) }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Requested Loan Amount</label>
                                    <input type="number" name="loan_amount" class="form-control" value="{{ form.get('loan_amount', 15000000) }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Loan Term (Years)</label>
                                    <input type="number" name="loan_term" class="form-control" value="{{ form.get('loan_term', 10) }}" min="1" max="40" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">CIBIL Score (300-900)</label>
                                    <input type="number" name="cibil_score" class="form-control" value="{{ form.get('cibil_score', 750) }}" min="300" max="900" required>
                                </div>
                            </div>

                            <!-- Column 3: Asset Details -->
                            <div class="col-md-4">
                                <span class="section-badge">3. Asset Portfolio</span>
                                <div class="mb-3">
                                    <label class="form-label">Residential Assets Value</label>
                                    <input type="number" name="residential_assets_value" class="form-control" value="{{ form.get('residential_assets_value', 4000000) }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Commercial Assets Value</label>
                                    <input type="number" name="commercial_assets_value" class="form-control" value="{{ form.get('commercial_assets_value', 2000000) }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Luxury Assets Value</label>
                                    <input type="number" name="luxury_assets_value" class="form-control" value="{{ form.get('luxury_assets_value', 10000000) }}" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Bank Asset Value</label>
                                    <input type="number" name="bank_asset_value" class="form-control" value="{{ form.get('bank_asset_value', 3000000) }}" required>
                                </div>
                            </div>
                        </div>

                        <div class="mt-4">
                            <button type="submit" class="btn btn-predict">Predict Loan Eligibility</button>
                        </div>
                    </form>

                    {% if prediction is not none %}
                        {% if prediction == 1 %}
                        <div class="result-box result-approved">
                            <h3>✅ Loan Approved!</h3>
                            <p class="mb-0">Approval Probability: <strong>{{ probability }}</strong></p>
                        </div>
                        {% else %}
                        <div class="result-box result-rejected">
                            <h3>❌ Loan Rejected</h3>
                            <p class="mb-0">Approval Probability: <strong>{{ probability }}</strong></p>
                        </div>
                        {% endif %}
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            error_msg="Model file 'model.pkl' not found. Make sure model.pkl is in the repository root directory.", 
            prediction=None, 
            form={}
        )

    prediction = None
    probability = None

    if request.method == "POST":
        form_data = request.form

        # Create DataFrame matching feature names
        input_data = pd.DataFrame([{
            'no_of_dependents': int(form_data['no_of_dependents']),
            'education': 1 if form_data['education'] == 'Graduate' else 0,
            'self_employed': 1 if form_data['self_employed'] == 'Yes' else 0,
            'income_annum': float(form_data['income_annum']),
            'loan_amount': float(form_data['loan_amount']),
            'loan_term': float(form_data['loan_term']),
            'cibil_score': float(form_data['cibil_score']),
            'residential_assets_value': float(form_data['residential_assets_value']),
            'commercial_assets_value': float(form_data['commercial_assets_value']),
            'luxury_assets_value': float(form_data['luxury_assets_value']),
            'bank_asset_value': float(form_data['bank_asset_value'])
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        prediction = int(pred)
        probability = f"{prob:.2%}"
        return render_template_string(HTML_TEMPLATE, prediction=prediction, probability=probability, form=form_data)

    return render_template_string(HTML_TEMPLATE, prediction=None, probability=None, form={})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
