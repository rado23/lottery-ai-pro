from flask import Flask, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fetcher import fetch_draws
from src.analyzer import analyze_draws
from src.predictor import generate_predictions
from src.ml_predictor import generate_ml_predictions

app = Flask(__name__)

# Fetch data when the app starts (once)
fetch_draws()

@app.route("/predict", methods=["GET"])
def predict():
    stats = analyze_draws()
    predictions = generate_predictions(stats)
    ml = generate_ml_predictions()

    return jsonify({
        "heuristic": predictions,
        "ml": ml
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
