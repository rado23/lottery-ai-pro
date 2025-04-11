# src/lotto/lotto_ml_predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from src.lotto.lotto_analyzer import analyze_lotto_draws

def number_frequency_features(df, main_cols):
    # One-hot encode the past draws
    mlb = MultiLabelBinarizer(classes=range(1, 60))  # Lotto: numbers 1-59
    X = mlb.fit_transform(df[main_cols].values.tolist())

    # Shift for supervised learning: predict the next draw from previous
    X = pd.DataFrame(X[:-1])
    y = df[main_cols].values[1:]
    y_df = pd.DataFrame(y)

    return X, y_df


def predict_lotto_with_ml(df, main_cols):

    # Clean + prepare
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if "ball" in col]

    X, y = number_frequency_features(df, main_cols)

    # Train model
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y)

    # Predict probabilities
    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    # Compute average probability for each number
    mean_probs = [p[0][1] if isinstance(p[0], (list, tuple)) else p[1] for p in probas]

    # Pick top 6 most probable numbers
    top_indices = sorted(range(len(mean_probs)), key=lambda i: mean_probs[i], reverse=True)[:6]
    top_numbers = [i + 1 for i in top_indices]  # +1 for 1-based Lotto numbers

    return {
        "main_numbers": sorted(top_numbers)
    }
