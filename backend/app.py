from flask import Flask, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import generate_predictions
from src.ml_predictor import generate_ml_predictions
from src.analyzer import analyze_draws



app = Flask(__name__)

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
