import os
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Auto-download CSV if missing ===
csv_path = "data/thunderball_draws.csv"
csv_url = "https://www.national-lottery.co.uk/results/thunderball/draw-history/csv"

if not os.path.exists(csv_path):
    os.makedirs("data", exist_ok=True)
    print("📥 Downloading Thunderball data...")
    df_download = pd.read_csv(csv_url)
    df_download.to_csv(csv_path, index=False)
    print("✅ Thunderball data saved.")

# === Load Data ===
df = pd.read_csv(csv_path)

main_cols = ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5']
thunder_col = 'Thunderball'


def number_frequency_features(df, main_cols, thunder_col):
    number_range = range(1, 40)
    thunder_range = range(1, 15)

    X = []
    y_main = [[] for _ in range(5)]
    y_thunder = []

    for i in range(len(df) - 1):
        current_draw = df.iloc[i]
        next_draw = df.iloc[i + 1]

        features = [0] * (len(number_range) + len(thunder_range))
        for col in main_cols:
            if pd.notna(current_draw[col]):
                idx = int(current_draw[col]) - 1
                features[idx] += 1

        if pd.notna(current_draw[thunder_col]):
            idx = int(current_draw[thunder_col]) - 1 + len(number_range)
            features[idx] += 1

        X.append(features)

        for j, col in enumerate(main_cols):
            y_main[j].append(int(next_draw[col]))

        y_thunder.append(int(next_draw[thunder_col]))

    return pd.DataFrame(X), pd.DataFrame(y_main).T, pd.Series(y_thunder)


def predict_thunderball_with_ml(df, main_cols, thunder_col):
    X, y_main, y_thunder = number_frequency_features(df, main_cols, thunder_col)

    # === Train multi-output classifier for main balls ===
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    X_train, X_val, y_train, y_val = train_test_split(X, y_main, test_size=0.15, random_state=42)
    model.fit(X_train, y_train)

    y_pred_main = model.predict(X_val)
    acc_main = accuracy_score(y_val, y_pred_main)
    print(f"✅ Main Numbers Model Accuracy: {acc_main:.3f}")

    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    # === Combine probabilities across positions ===
    number_scores = {}
    for i in range(5):
        for idx, prob in enumerate(probas[i][0]):
            number = idx + 1
            number_scores[number] = number_scores.get(number, 0) + prob

    # Sort and select top 5 unique
    top_5 = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
    main_preds = []
    seen = set()
    total_conf = 0
    for num, score in top_5:
        if num not in seen:
            main_preds.append(num)
            seen.add(num)
            total_conf += score
        if len(main_preds) == 5:
            break

    avg_main_conf = round(total_conf / 5, 4)

    # === Train Thunderball model ===
    thunder_model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train_th, X_val_th, y_train_th, y_val_th = train_test_split(X, y_thunder, test_size=0.15, random_state=42)
    thunder_model.fit(X_train_th, y_train_th)
    y_pred_th = thunder_model.predict(X_val_th)
    acc_th = accuracy_score(y_val_th, y_pred_th)
    print(f"✅ Thunderball Model Accuracy: {acc_th:.3f}")

    thunder_pred = int(thunder_model.predict(last_features)[0])
    prob_th = max(thunder_model.predict_proba(last_features)[0])
    avg_thunder_conf = round(prob_th, 4)

    return {
        "main_numbers": sorted(main_preds),
        "thunderball": thunder_pred,
        "confidence": {
            "main_numbers": avg_main_conf,
            "thunderball": avg_thunder_conf
        }
    }


# === Manual test ===
if __name__ == "__main__":
    result = predict_thunderball_with_ml(df, main_cols, thunder_col)
    print("🔮 Thunderball ML Prediction:")
    print("Main Numbers:", result["main_numbers"])
    print("Thunderball:", result["thunderball"])
    print("Confidence:", result["confidence"])
