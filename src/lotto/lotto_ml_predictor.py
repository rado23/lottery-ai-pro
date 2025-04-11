# src/lotto/lotto_ml_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def number_frequency_features(df, main_cols):
    """
    Converts previous draw results into binary feature matrix for training.
    Each row represents one draw, with 59 binary columns (1-59 Lotto numbers).
    """
    mlb = MultiLabelBinarizer(classes=range(1, 60))  # Lotto numbers: 1–59
    one_hot = mlb.fit_transform(df[main_cols].values.tolist())

    # Predict the next draw from the current one
    X = pd.DataFrame(one_hot[:-1])
    y = pd.DataFrame(one_hot[1:], columns=[f"n_{i+1}" for i in range(59)])

    return X, y


def predict_lotto_with_ml(df, main_cols):
    """
    Trains a model to predict Lotto draws using historical data.
    Uses binary classification for each number (1 if drawn, 0 otherwise).
    """
    # Ensure clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if "ball" in col]

    X, y = number_frequency_features(df, main_cols)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    model.fit(X, y)

    # Predict next draw from last known draw
    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    # Extract probabilities for each number being in next draw
    number_probs = []
    for idx, prob in enumerate(probas):
        if len(prob[0]) == 2:
            number_probs.append((idx + 1, prob[0][1]))  # probability of "1"
        else:
            number_probs.append((idx + 1, 0))  # fallback for unseen labels

    # Sort and return top 6 most likely numbers
    top_numbers = sorted(number_probs, key=lambda x: x[1], reverse=True)[:6]
    selected_numbers = sorted([n for n, _ in top_numbers])

    return {
        "main_numbers": selected_numbers
    }
