
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
import random

# Load draw data
df = pd.read_csv("data/thunderball_draws.csv")

# Clean and normalize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Identify main and thunderball columns
main_cols = ["ball_1", "ball_2", "ball_3", "ball_4", "ball_5"]
thunder_col = "thunderball"

# Prepare features: each row is the previous draw's numbers (frequency)
def number_frequency_features(df, main_cols, thunder_col):
    feature_rows = []
    label_main = [[] for _ in range(5)]
    label_thunder = []

    for i in range(5, len(df)):
        prev = df.iloc[i-5:i]
        next_draw = df.iloc[i]

        main_flat = prev[main_cols].values.flatten()
        thunder_values = prev[thunder_col].values

        main_counts = Counter(main_flat)
        thunder_counts = Counter(thunder_values)

        row = []
        for n in range(1, 40):  # 1-39
            row.append(main_counts.get(n, 0))
        for t in range(1, 15):  # 1-14
            row.append(thunder_counts.get(t, 0))

        feature_rows.append(row)

        for j, col in enumerate(main_cols):
            label_main[j].append(next_draw[col])
        label_thunder.append(next_draw[thunder_col])

    X = np.array(feature_rows)
    y_main = [np.array(lab) - 1 for lab in label_main]
    y_thunder = np.array(label_thunder) - 1

    return X, y_main, y_thunder

# Build training set
X, y_main, y_thunder = number_frequency_features(df, main_cols, thunder_col)

# Train one model per main number position
main_models = []
for y in y_main:
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", num_class=39)
    # Hack to ensure all classes are present for 0–38
    y_padded = np.concatenate([y, np.arange(39)])
    X_padded = np.concatenate([X, np.tile(X[0], (39, 1))])  # Repeat a dummy row

    model.fit(X_padded, y_padded)

    main_models.append(model)

# Train Thunderball model
thunder_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", num_class=14)
y_thunder_padded = np.concatenate([y_thunder, np.arange(14)])
X_thunder_padded = np.concatenate([X, np.tile(X[0], (14, 1))])

thunder_model.fit(X_thunder_padded, y_thunder_padded)


# Make a prediction using the latest 5 draws
def predict_thunderball_with_ml(df, main_cols, thunder_col):
    recent = df.iloc[-5:]
    main_flat = recent[main_cols].values.flatten()
    thunder_values = recent[thunder_col].values

    main_counts = Counter(main_flat)
    thunder_counts = Counter(thunder_values)

    row = []
    for n in range(1, 40):
        row.append(main_counts.get(n, 0))
    for t in range(1, 15):
        row.append(thunder_counts.get(t, 0))

    input_vector = np.array(row).reshape(1, -1)

    predicted_main = []
    for model in main_models:
        pred = int(model.predict(input_vector)[0]) + 1
        while pred in predicted_main:
            pred = random.randint(1, 39)
        predicted_main.append(pred)

    predicted_thunder = int(thunder_model.predict(input_vector)[0]) + 1
    return {
        "main": sorted(predicted_main),
        "thunderball": predicted_thunder
    }

# Predict and show result
if __name__ == "__main__":
    prediction = predict_thunderball_with_ml(df, main_cols, thunder_col)
    print(prediction)
