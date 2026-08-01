import os
import numpy as np
import xgboost as xgb
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Resolve absolute path for Vercel serverless environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.xgb')
FALLBACK_PATH = os.path.join(BASE_DIR, 'Xgboost_model.pkl')

booster = None

def get_booster():
    """Lazy loader to prevent startup crash if model loading takes extra time."""
    global booster
    if booster is None:
        booster = xgb.Booster()
        if os.path.exists(MODEL_PATH):
            booster.load_model(MODEL_PATH)
        elif os.path.exists(FALLBACK_PATH):
            booster.load_model(FALLBACK_PATH)
        else:
            raise FileNotFoundError("Model file not found. Ensure model.xgb or model.pkl exists.")
    return booster

# Default feature names fallback
DEFAULT_FEATURES = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
]

def get_feature_names():
    try:
        model = get_booster()
        return model.feature_names or DEFAULT_FEATURES
    except Exception:
        return DEFAULT_FEATURES

# HTML UI Template
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
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .card {
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }
        .form-control, .form-select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f8fafc;
            border-radius: 8px;
        }
        .form-control:focus, .form-select:focus {
            background: rgba(15, 23, 42, 0.8);
            color: #fff;
            border-color: #3b82f6;
            box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25);
        }
        .btn-primary {
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            border: none;
            padding: 12px;
            font-weight: 600;
            border-radius: 8px;
        }
        .result-badge {
            font-size: 1.25rem;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card p-4 p-md-5">
                    <h2 class="text-center mb-1 text-primary">Loan Approval Predictor</h2>
                    <p class="text-center text-muted mb-4">ML Serverless Deployment</p>
                    
                    <form id="predictForm">
                        <div class="row g-3">
                            {% for feature in features %}
                            <div class="col-md-6">
                                <label for="{{ feature }}" class="form-label text-capitalize">
                                    {{ feature.replace('_', ' ') }}
                                </label>
                                {% if feature in ['education', 'self_employed'] %}
                                <select class="form-select" id="{{ feature }}" name="{{ feature }}" required>
                                    <option value="0">Graduate / No (0)</option>
                                    <option value="1">Not Graduate / Yes (1)</option>
                                </select>
                                {% else %}
                                <input type="number" step="any" class="form-control" id="{{ feature }}" name="{{ feature }}" placeholder="0" required>
                                {% endif %}
                            </div>
                            {% endfor %}
                        </div>

                        <button type="submit" class="btn btn-primary w-100 mt-4" id="submitBtn">
                            Predict Outcome
                        </button>
                    </form>

                    <div id="resultBox" class="result-badge"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = document.getElementById('submitBtn');
            const resultBox = document.getElementById('resultBox');
            
            submitBtn.innerText = "Processing Prediction...";
            submitBtn.disabled = true;

            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((value, key) => { data[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                resultBox.style.display = 'block';
                
                if (result.status === 'success') {
                    const probPercent = (result.probability * 100).toFixed(2);
                    if (result.prediction === 1) {
                        resultBox.className = 'result-badge bg-success bg-opacity-25 text-success border border-success';
                        resultBox.innerHTML = `<strong>Loan Approved</strong><br>Confidence Score: ${probPercent}%`;
                    } else {
                        resultBox.className = 'result-badge bg-danger bg-opacity-25 text-danger border border-danger';
                        resultBox.innerHTML = `<strong>Loan Rejected</strong><br>Confidence Score: ${probPercent}%`;
                    }
                } else {
                    resultBox.className = 'result-badge bg-warning bg-opacity-25 text-warning border border-warning';
                    resultBox.innerText = 'Error: ' + result.message;
                }
            } catch (err) {
                resultBox.style.display = 'block';
                resultBox.className = 'result-badge bg-danger bg-opacity-25 text-danger border border-danger';
                resultBox.innerText = 'Failed to communicate with server.';
            } finally {
                submitBtn.innerText = "Predict Outcome";
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    features = get_feature_names()
    return render_template_string(HTML_TEMPLATE, features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = get_feature_names()
        
        feature_values = [data[feat] for feat in features]
        dmatrix = xgb.DMatrix(np.array([feature_values]), feature_names=features)
        
        model = get_booster()
        probs = model.predict(dmatrix)
        probability = float(probs[0])
        prediction = int(probability >= 0.5)

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Serverless entry point
app_instance = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
