from flask import Flask, jsonify
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml
from src.thunderball.thunderball_predictor import generate_thunderball_predictions
from src.thunderball.thunderball_ml_predictor import predict_thunderball_with_ml
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
from src.euromillions.euromillions_predictor import generate_euromillions_predictions

app = Flask(__name__)

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
    heuristic = generate_thunderball_predictions()
    return jsonify({"heuristic": heuristic})

from src.thunderball.thunderball_analyzer import analyze_thunderball_draws

@app.route("/predict/thunderball-ml", methods=["GET"])
def predict_thunderball_ml():
    df, main_cols, thunder_col = analyze_thunderball_draws()
    ml = predict_thunderball_with_ml(df, main_cols, thunder_col)
    return jsonify({"ml": ml})


if __name__ == "__main__":
    app.run(debug=True)
