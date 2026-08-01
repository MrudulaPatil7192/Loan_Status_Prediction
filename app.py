import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Resolve absolute path to the model file for Vercel execution context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --input-bg: #0f172a;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --success-color: #10b981;
            --danger-color: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 2rem 1rem; }
        .container { background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 16px; width: 100%; max-width: 800px; padding: 2.5rem; }
        .header { text-align: center; margin-bottom: 2rem; }
        .header h1 { font-size: 2rem; font-weight: 700; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
        .header p { color: var(--text-secondary); font-size: 0.95rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; }
        .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .form-group label { font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
        .form-group input, .form-group select { background-color: var(--input-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 0.75rem 1rem; color: var(--text-primary); font-size: 0.95rem; outline: none; }
        .submit-btn { grid-column: 1 / -1; margin-top: 1rem; background: var(--accent-gradient); color: white; border: none; border-radius: 8px; padding: 1rem; font-size: 1rem; font-weight: 600; cursor: pointer; }
        .result-box { margin-top: 2rem; padding: 1.25rem; border-radius: 8px; text-align: center; font-weight: 600; font-size: 1.1rem; }
        .result-box.approved { background-color: rgba(16, 185, 129, 0.15); border: 1px solid var(--success-color); color: var(--success-color); }
        .result-box.rejected { background-color: rgba(239, 68, 68, 0.15); border: 1px solid var(--danger-color); color: var(--danger-color); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Loan Approval Predictor</h1>
            <p>Enter financial and asset details to evaluate loan eligibility</p>
        </div>
        <form method="POST" action="/predict">
            <div class="form-grid">
                <div class="form-group"><label>Dependents</label><input type="number" name="no_of_dependents" min="0" required placeholder="e.g. 2"></div>
                <div class="form-group"><label>Education</label><select name="education" required><option value="1">Graduate</option><option value="0">Not Graduate</option></select></div>
                <div class="form-group"><label>Self Employed</label><select name="self_employed" required><option value="0">No</option><option value="1">Yes</option></select></div>
                <div class="form-group"><label>Annual Income ($)</label><input type="number" name="income_annum" step="any" required placeholder="e.g. 5000000"></div>
                <div class="form-group"><label>Loan Amount ($)</label><input type="number" name="loan_amount" step="any" required placeholder="e.g. 15000000"></div>
                <div class="form-group"><label>Loan Term (Years)</label><input type="number" name="loan_term" min="1" required placeholder="e.g. 10"></div>
                <div class="form-group"><label>CIBIL Score</label><input type="number" name="cibil_score" min="300" max="900" required placeholder="e.g. 750"></div>
                <div class="form-group"><label>Residential Assets ($)</label><input type="number" name="residential_assets_value" step="any" required placeholder="e.g. 2000000"></div>
                <div class="form-group"><label>Commercial Assets ($)</label><input type="number" name="commercial_assets_value" step="any" required placeholder="e.g. 4000000"></div>
                <div class="form-group"><label>Luxury Assets ($)</label><input type="number" name="luxury_assets_value" step="any" required placeholder="e.g. 8000000"></div>
                <div class="form-group"><label>Bank Assets ($)</label><input type="number" name="bank_asset_value" step="any" required placeholder="e.g. 3000000"></div>
                <button type="submit" class="submit-btn">Run Prediction</button>
            </div>
        </form>
        {% if prediction_text %}
            <div class="result-box {{ result_class }}">{{ prediction_text }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text="Error: Model file 'xgboost_model.pkl' not loaded correctly.", 
            result_class="rejected"
        )

    try:
        features = [
            float(request.form["no_of_dependents"]),
            float(request.form["education"]),
            float(request.form["self_employed"]),
            float(request.form["income_annum"]),
            float(request.form["loan_amount"]),
            float(request.form["loan_term"]),
            float(request.form["cibil_score"]),
            float(request.form["residential_assets_value"]),
            float(request.form["commercial_assets_value"]),
            float(request.form["luxury_assets_value"]),
            float(request.form["bank_asset_value"])
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]

        result_text = "Loan Approved" if prediction == 1 else "Loan Rejected"
        result_class = "approved" if prediction == 1 else "rejected"

        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Prediction Result: {result_text}", 
            result_class=result_class
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Error: {str(e)}", 
            result_class="rejected"
        )

# Required entry point for Vercel WSGI
app = app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
