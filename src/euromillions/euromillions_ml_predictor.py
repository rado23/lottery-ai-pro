# src/euromillions_ml_predictor.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def create_label_mapping(values):
    sorted_unique = sorted(set(values))
    label_map = {val: i for i, val in enumerate(sorted_unique)}
    reverse_map = {i: val for val, i in label_map.items()}
    return label_map, reverse_map

def build_training_data():
    df = pd.read_csv("data/euromillions_draws.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = df["date"].astype(int) // 10**9  # UNIX timestamp

    features = []
    labels_main = [[] for _ in range(5)]
    labels_star = [[] for _ in range(2)]

    for _, row in df.iterrows():
        f = [
            row["timestamp"],
            sum(row[f"main_{i+1}"] for i in range(5)),
            sum(row[f"star_{i+1}"] for i in range(2))
        ]
        features.append(f)

        for i in range(5):
            labels_main[i].append(row[f"main_{i+1}"])
        for i in range(2):
            labels_star[i].append(row[f"star_{i+1}"])

    X = np.array(features)
    return X, labels_main, labels_star

def train_models(X, y_list, label_type):
    models = []
    for idx, y_raw in enumerate(y_list):
        X_train, X_val, y_train_raw, y_val_raw = train_test_split(X, y_raw, test_size=0.15, random_state=42)

        # Create label map only for the training set
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

        # Filter out -1 values from accuracy evaluation
        valid_indices = y_val != -1
        y_val = y_val[valid_indices]
        y_pred = y_pred[valid_indices]

        acc = accuracy_score(y_val, y_pred)
        print(f"✅ {label_type}_{idx+1} Accuracy: {acc:.3f}")
        models.append((model, reverse_map))
    return models


def predict_draw(models, X_sample, k):
    all_probs = []
    for model, reverse_map in models:
        probas = model.predict_proba(X_sample)[0]
        probs_with_labels = [(reverse_map[j], probas[j]) for j in range(len(probas))]
        all_probs.extend(probs_with_labels)

    # Sort all by probability
    all_probs.sort(key=lambda x: x[1], reverse=True)

    # Pick top k unique values
    unique = []
    seen = set()
    confidence = []
    for value, prob in all_probs:
        if value not in seen:
            seen.add(value)
            unique.append(value)
            confidence.append(prob)
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

    avg_main_conf = round(float(sum(main_conf)) / len(main_conf), 4)
    avg_star_conf = round(float(sum(star_conf)) / len(star_conf), 4)

    return {
        "main_numbers": sorted(int(n) for n in main_pred),
        "lucky_stars": sorted(int(s) for s in star_pred),
        "confidence": {
            "main_numbers": round(float(sum(main_conf)) / len(main_conf), 4),
            "lucky_stars": round(float(sum(star_conf)) / len(star_conf), 4)
        }
    }

