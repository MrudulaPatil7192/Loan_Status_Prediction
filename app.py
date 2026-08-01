import os
import numpy as np
import xgboost as xgb
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the XGBoost Booster model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.xgb')

def load_model():
    model = xgb.Booster()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    else:
        # Fallback to model.pkl if saved with pickle extension
        fallback = os.path.join(os.path.dirname(__file__), 'Xgboost_model.pkl')
        if os.path.exists(fallback):
            model.load_model(fallback)
        else:
            raise FileNotFoundError("Please save your model data as 'model.xgb' in the app directory.")
    return model

try:
    booster = load_model()
    # Extract feature names directly from the model metadata if present
    FEATURE_NAMES = booster.feature_names or [
        "no_of_dependents", "education", "self_employed", "income_annum",
        "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]
except Exception as e:
    print(f"Error loading model: {e}")
    FEATURE_NAMES = [
        "no_of_dependents", "education", "self_employed", "income_annum",
        "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]

# HTML/CSS/JS Template for modern, interactive user interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Prediction System</title>
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
            background: rgba(30, 41, 59, 0.7);
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
            transition: transform 0.2s;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
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
                    <h2 class="text-center mb-1 text-primary">Loan Eligibility Evaluator</h2>
                    <p class="text-center text-muted mb-4">Interactive XGBoost Model Interface</p>
                    
                    <form id="predictForm">
                        <div class="row g-3">
                            {% for feature in features %}
                            <div class="col-md-6">
                                <label for="{{ feature }}" class="form-label text-capitalize">
                                    {{ feature.replace('_', ' ') }}
                                </label>
                                {% if feature in ['education', 'self_employed'] %}
                                <select class="form-select" id="{{ feature }}" name="{{ feature }}" required>
                                    <option value="0">No / Graduate (0)</option>
                                    <option value="1">Yes / Not Graduate (1)</option>
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
                
                if (result.status === 'success') {
                    const probPercent = (result.probability * 100).toFixed(2);
                    resultBox.style.display = 'block';
                    
                    if (result.prediction === 1) {
                        resultBox.className = 'result-badge bg-success bg-opacity-25 text-success border border-success';
                        resultBox.innerHTML = `<strong>Approved / High Confidence</strong><br>Probability Score: ${probPercent}%`;
                    } else {
                        resultBox.className = 'result-badge bg-danger bg-opacity-25 text-danger border border-danger';
                        resultBox.innerHTML = `<strong>Rejected / Low Confidence</strong><br>Probability Score: ${probPercent}%`;
                    }
                } else {
                    resultBox.style.display = 'block';
                    resultBox.className = 'result-badge bg-warning bg-opacity-25 text-warning border border-warning';
                    resultBox.innerText = 'Error: ' + result.message;
                }
            } catch (err) {
                resultBox.style.display = 'block';
                resultBox.className = 'result-badge bg-danger bg-opacity-25 text-danger border border-danger';
                resultBox.innerText = 'Failed to fetch response from server.';
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
    return render_template_string(HTML_TEMPLATE, features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare vector in the exact feature sequence expected by the model
        feature_values = [data[feat] for feat in FEATURE_NAMES]
        dmatrix = xgb.DMatrix(np.array([feature_values]), feature_names=FEATURE_NAMES)
        
        # Perform prediction
        probs = booster.predict(dmatrix)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
