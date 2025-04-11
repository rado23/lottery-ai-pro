# backend/app.py

from flask import Flask, jsonify
from flask_cors import CORS

# --- Thunderball imports ---
from src.thunderball.download_thunderball_csv import save_thunderball_draws_csv
from src.thunderball.thunderball_analyzer import analyze_thunderball_draws
from src.thunderball.thunderball_ml_predictor import predict_thunderball_with_ml

# --- EuroMillions imports ---
from src.euromillions.euromillions_fetcher import fetch_draws  # ✅ updated to match restored version
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml

import os

app = Flask(__name__)
CORS(app)

# Initial CSV generation
save_thunderball_draws_csv()
fetch_draws()  # ✅ updated function for EuroMillions

# --- EuroMillions Endpoints ---
@app.route("/predict/euromillions", methods=["GET"])
def predict_euromillions():
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
    stats = analyze_thunderball_draws()
    heuristic = generate_thunderball_predictions(stats)
    return jsonify({"heuristic": heuristic})


@app.route("/predict/thunderball-ml", methods=["GET"])
def predict_thunderball_ml():
    ml = predict_thunderball_with_ml()
    return jsonify({"ml": ml})


# --- Run ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
