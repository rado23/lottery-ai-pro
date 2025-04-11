import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from src.setforlife.setforlife_analyzer import analyze_setforlife_draws

def number_frequency_features(df, main_cols, life_col):
    mlb = MultiLabelBinarizer(classes=range(1, 48))  # 1-47 for main balls
    X = mlb.fit_transform(df[main_cols].values.tolist())

    X = pd.DataFrame(X[:-1])
    y_main = df[main_cols].values[1:]
    y_main_df = pd.DataFrame(y_main)

    y_life = df[life_col].apply(lambda x: int(re.search(r"\d+", str(x)).group()) if pd.notna(x) else 0).values[1:]

    return X, y_main_df, y_life

def predict_setforlife_with_ml(df, main_cols, life_col):
    X, y_main, y_life = number_frequency_features(df, main_cols, life_col)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y_main)
    last_features = X.iloc[[-1]]
    main_preds = model.predict(last_features)[0].tolist()

    life_model = RandomForestClassifier(n_estimators=100, random_state=42)
    life_model.fit(X, y_life)
    life_pred = int(life_model.predict(last_features)[0])

    return {
        "main_numbers": sorted(main_preds),
        "life_ball": life_pred
    }
