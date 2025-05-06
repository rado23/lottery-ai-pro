# src/euromillions_ml_predictor.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def build_training_data():
    df = pd.read_csv("data/euromillions_draws.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = df["date"].astype(int) // 10**9  # UNIX timestamp

    features = []
    labels_main = [[] for _ in range(5)]
    labels_star = [[] for _ in range(2)]

    for _, row in df.iterrows():
        f = [
            int(row["timestamp"]),
            int(sum(row[f"main_{i+1}"] for i in range(5))),
            int(sum(row[f"star_{i+1}"] for i in range(2)))
        ]
        features.append(f)

        for i in range(5):
            labels_main[i].append(int(row[f"main_{i+1}"]))
        for i in range(2):
            labels_star[i].append(int(row[f"star_{i+1}"]))

    X = np.array(features)
    return X, labels_main, labels_star

def train_models(X, y_list, label_type):
    models = []
    for idx, y_raw in enumerate(y_list):
        X_train, X_val, y_train_raw, y_val_raw = train_test_split(X, y_raw, test_size=0.15, random_state=42)

        label_map = {val: i for i, val in enumerate(sorted(set(y_train_raw)))}
        reverse_map = {i: val for val, i in label_map.items()}

        y_train = np.array([label_map[val] for val in y_train_raw])
        y_val = np.array([label_map.get(val, -1) for val in y_val_raw])

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='mlogloss',
            num_class=len(label_map)
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        valid_indices = y_val != -1
        acc = accuracy_score(y_val[valid_indices], y_pred[valid_indices])
        print(f"✅ {label_type}_{idx+1} Accuracy: {acc:.3f}")
        models.append((model, reverse_map))
    return models

def predict_draw(models, X_sample, k):
    all_probs = []
    for model, reverse_map in models:
        probas = model.predict_proba(X_sample)[0]
        probs_with_labels = [(int(reverse_map[j]), float(probas[j])) for j in range(len(probas))]
        all_probs.extend(probs_with_labels)

    all_probs.sort(key=lambda x: x[1], reverse=True)

    unique = []
    seen = set()
    confidence = []
    for value, prob in all_probs:
        if value not in seen:
            seen.add(value)
            unique.append(int(value))
            confidence.append(float(prob))
        if len(unique) == k:
            break
    return unique, confidence

def predict_euromillions_with_ml():
    X, y_main, y_star = build_training_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = X_scaled[:-1]
    X_pred = X_scaled[-1].reshape(1, -1)

    y_main_train = [y[:-1] for y in y_main]
    y_star_train = [y[:-1] for y in y_star]

    main_models = train_models(X_train, y_main_train, label_type="Main")
    star_models = train_models(X_train, y_star_train, label_type="Star")

    print("🔮 Generating ML prediction...")

    main_pred, main_conf = predict_draw(main_models, X_pred, k=5)
    star_pred, star_conf = predict_draw(star_models, X_pred, k=2)

    return {
        "main_numbers": sorted(main_pred),
        "lucky_stars": sorted(star_pred),
        "confidence": {
            "main_numbers": round(sum(main_conf) / len(main_conf), 4),
            "lucky_stars": round(sum(star_conf) / len(star_conf), 4)
        }
    }
