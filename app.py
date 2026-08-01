from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None

# Single-page HTML interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }
        .container { max-width: 800px; background: white; margin: auto; padding: 25px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-weight: bold; margin-bottom: 5px; color: #555; }
        input, select { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
        button { grid-column: span 2; padding: 12px; background: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 15px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; text-align: center; font-size: 18px; font-weight: bold; }
        .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .danger { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏦 Loan Approval Prediction System</h2>
        {% if error %}
            <div class="result danger">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/">
            <div class="grid">
                <div class="form-group">
                    <label>Number of Dependents</label>
                    <input type="number" name="no_of_dependents" value="{{ form.get('no_of_dependents', 2) }}" min="0" max="20" required>
                </div>
                <div class="form-group">
                    <label>Education</label>
                    <select name="education">
                        <option value="Graduate" {% if form.get('education') == 'Graduate' %}selected{% endif %}>Graduate</option>
                        <option value="Not Graduate" {% if form.get('education') == 'Not Graduate' %}selected{% endif %}>Not Graduate</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Self Employed</label>
                    <select name="self_employed">
                        <option value="No" {% if form.get('self_employed') == 'No' %}selected{% endif %}>No</option>
                        <option value="Yes" {% if form.get('self_employed') == 'Yes' %}selected{% endif %}>Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Annual Income ($)</label>
                    <input type="number" name="income_annum" value="{{ form.get('income_annum', 5000000) }}" required>
                </div>
                <div class="form-group">
                    <label>Loan Amount ($)</label>
                    <input type="number" name="loan_amount" value="{{ form.get('loan_amount', 15000000) }}" required>
                </div>
                <div class="form-group">
                    <label>Loan Term (Years)</label>
                    <input type="number" name="loan_term" value="{{ form.get('loan_term', 10) }}" required>
                </div>
                <div class="form-group">
                    <label>CIBIL Score (300 - 900)</label>
                    <input type="number" name="cibil_score" value="{{ form.get('cibil_score', 750) }}" min="300" max="900" required>
                </div>
                <div class="form-group">
                    <label>Residential Assets Value ($)</label>
                    <input type="number" name="residential_assets_value" value="{{ form.get('residential_assets_value', 4000000) }}" required>
                </div>
                <div class="form-group">
                    <label>Commercial Assets Value ($)</label>
                    <input type="number" name="commercial_assets_value" value="{{ form.get('commercial_assets_value', 2000000) }}" required>
                </div>
                <div class="form-group">
                    <label>Luxury Assets Value ($)</label>
                    <input type="number" name="luxury_assets_value" value="{{ form.get('luxury_assets_value', 10000000) }}" required>
                </div>
                <div class="form-group" style="grid-column: span 2;">
                    <label>Bank Asset Value ($)</label>
                    <input type="number" name="bank_asset_value" value="{{ form.get('bank_asset_value', 3000000) }}" required>
                </div>
                <button type="submit">Predict Loan Status</button>
            </div>
        </form>

        {% if prediction is not none %}
            {% if prediction == 1 %}
                <div class="result success">✅ Loan Approved! (Probability: {{ probability }})</div>
            {% else %}
                <div class="result danger">❌ Loan Rejected. (Probability: {{ probability }})</div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Error: model.pkl file not found in root directory.", prediction=None, form={})

    prediction = None
    probability = None

    if request.method == "POST":
        form_data = request.form
        
        # Format input features for XGBoost model
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

# Top-level 'app' object required by Vercel
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
