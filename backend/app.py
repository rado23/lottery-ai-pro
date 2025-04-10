import os
from flask import Flask, jsonify
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml
from src.euromillions.euromillions_predictor import generate_euromillions_predictions
from src.thunderball.thunderball_predictor import generate_thunderball_predictions
from src.thunderball.thunderball_ml_predictor import predict_thunderball_with_ml
import pandas as pd

app = Flask(__name__)

# --- EuroMillions Endpoints ---
@app.route("/predict/euromillions", methods=["GET"])
def predict_euromillions():
    from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
    stats = analyze_euromillions_draws()
    heuristic = generate_euromillions_predictions(stats)
    return jsonify({"heuristic": heuristic})

@app.route("/predict/euromillions-ml", methods=["GET"])
def predict_euromillions_ml():
    ml = predict_euromillions_with_ml()
    return jsonify({"ml": ml})

# --- Thunderball Endpoints ---
@app.route("/predict/thunderball", methods=["GET"])
def predict_thunderball():
    heuristic = generate_thunderball_predictions()
    return jsonify({"heuristic": heuristic})

@app.route("/predict/thunderball-ml", methods=["GET"])
def predict_thunderball_ml():
    from src.thunderball.thunderball_ml_predictor import df, main_cols, thunder_col
    ml = predict_thunderball_with_ml(df, main_cols, thunder_col)
    return jsonify({"ml": ml})

# --- Start ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
