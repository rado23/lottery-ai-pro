# backend/app.py

from flask import Flask, jsonify
from flask_cors import CORS
import os

# --- EuroMillions ---
from src.euromillions.euromillions_fetcher import fetch_draws
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml
from src.euromillions.euromillions_predictor import generate_euromillions_predictions

# --- Thunderball ---
from src.thunderball.download_thunderball_csv import save_thunderball_draws_csv
from src.thunderball.thunderball_analyzer import analyze_thunderball_draws
from src.thunderball.thunderball_ml_predictor import predict_thunderball_with_ml
from src.thunderball.thunderball_predictor import generate_thunderball_predictions

# 🧠 Load data on start
save_thunderball_draws_csv()
fetch_draws()

# Initialize app
app = Flask(__name__)
CORS(app)


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
    df, main_cols, thunder_col = analyze_thunderball_draws()
    stats = {"df": df, "main_cols": main_cols, "thunder_col": thunder_col}
    heuristic = generate_thunderball_predictions(stats)
    return jsonify({"heuristic": heuristic})


@app.route("/predict/thunderball-ml", methods=["GET"])
def predict_thunderball_ml():
    df, main_cols, thunder_col = analyze_thunderball_draws()
    ml = predict_thunderball_with_ml(df, main_cols, thunder_col)
    return jsonify({"ml": ml})


# --- Start the server (Render sets the PORT env variable) ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
