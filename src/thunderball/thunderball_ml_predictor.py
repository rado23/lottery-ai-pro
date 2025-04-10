import os
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    number_range = range(1, 40)  # 1–39 for main numbers
    thunder_range = range(1, 15)  # 1–14 for Thunderball

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

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y_main)
    last_features = X.iloc[[-1]]
    main_preds = model.predict(last_features)[0].tolist()

    thunder_model = RandomForestClassifier(n_estimators=100, random_state=42)
    thunder_model.fit(X, y_thunder)
    thunder_pred = thunder_model.predict(last_features)[0]

    return {
        "main_numbers": sorted(main_preds),
        "thunderball": thunder_pred
    }


# === For testing directly ===
if __name__ == "__main__":
    result = predict_thunderball_with_ml(df, main_cols, thunder_col)
    print("🔮 Thunderball ML Prediction:")
    print("Main Numbers:", result["main_numbers"])
    print("Thunderball:", result["thunderball"])
