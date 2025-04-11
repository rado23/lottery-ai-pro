from flask import Flask, jsonify
from flask_cors import CORS

# --- EuroMillions Imports ---
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml
from src.euromillions.euromillions_predictor import generate_euromillions_predictions
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
from src.euromillions.euromillions_fetcher import save_euromillions_draws_csv

# --- Thunderball Imports ---
from src.thunderball.thunderball_ml_predictor import predict_thunderball_with_ml
from src.thunderball.thunderball_predictor import generate_thunderball_predictions
from src.thunderball.thunderball_analyzer import analyze_thunderball_draws
from src.thunderball.download_thunderball_csv import save_thunderball_draws_csv

# --- Setup ---
app = Flask(__name__)
CORS(app)

# Download data at startup
print("📥 Downloading EuroMillions data...")
save_euromillions_draws_csv()

print("📥 Downloading Thunderball data...")
save_thunderball_draws_csv()


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


if __name__ == "__main__":
    app.run(debug=True)
