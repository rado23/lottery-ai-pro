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

# --- Lotto ---
from src.lotto.download_lotto_csv import save_lotto_draws_csv
from src.lotto.lotto_predictor import generate_lotto_predictions
from src.lotto.lotto_analyzer import analyze_lotto_draws
from src.lotto.lotto_ml_predictor import predict_lotto_with_ml

# --- Set For Life ---
from src.setforlife.download_setforlife_csv import save_setforlife_draws_csv
from src.setforlife.setforlife_predictor import generate_setforlife_predictions
from src.setforlife.setforlife_analyzer import analyze_setforlife_draws
from src.setforlife.setforlife_ml_predictor import predict_setforlife_with_ml

# 🧠 Load data on start
save_thunderball_draws_csv()
fetch_draws()
save_lotto_draws_csv()
save_setforlife_draws_csv()

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

# --- Lotto Endpoints ---
@app.route("/predict/lotto", methods=["GET"])
def predict_lotto():
    stats = analyze_lotto_draws()
    heuristic = generate_lotto_predictions(stats)
    return jsonify({"heuristic": heuristic})


@app.route("/predict/lotto-ml", methods=["GET"])
def predict_lotto_ml():
    df, main_cols = analyze_lotto_draws(return_raw=True)
    ml = predict_lotto_with_ml(df, main_cols)
    return jsonify({"ml": ml})

# --- Set for Life Endpoints ---
@app.route("/predict/setforlife", methods=["GET"])
def predict_setforlife():
    stats = analyze_setforlife_draws()
    heuristic = generate_setforlife_predictions(stats)
    return jsonify({"heuristic": heuristic})


@app.route("/predict/setforlife-ml", methods=["GET"])
def predict_setforlife_ml():
    df, main_cols, life_col = analyze_setforlife_draws(return_raw=True)
    ml = predict_setforlife_with_ml(df, main_cols, life_col)
    return jsonify({"ml": ml})


# --- Start the server (Render sets the PORT env variable) ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
