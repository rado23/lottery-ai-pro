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
    stats = analyze_lotto_draws()
    df = pd.read_csv("data/lotto_draws.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if "ball" in col]

    X, y = number_frequency_features(df, main_cols)

    # Use MultiOutputClassifier to predict 6 numbers
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y)

    # Predict next draw using last known feature row
    last_features = X.iloc[[-1]]
    predicted = model.predict(last_features)[0]

    return {
        "main_numbers": sorted([int(n) for n in predicted])
    }
