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

    y_main = df[main_cols].applymap(
        lambda x: int(re.search(r"\d+", str(x)).group()) if pd.notna(x) else 0
    ).values[1:]
    y_main_df = pd.DataFrame(y_main)

    y_life = df[life_col].apply(
        lambda x: int(re.search(r"\d+", str(x)).group()) if pd.notna(x) else 0
    ).values[1:]

    return X, y_main_df, y_life


def predict_setforlife_with_ml(df, main_cols, life_col):
    # Defensive filtering
    main_cols = [col for col in df.columns if re.fullmatch(r"ball_[1-5]", col)]
    life_col = next((col for col in df.columns if "life" in col.lower()), None)
    if not life_col:
        raise ValueError("Life Ball column not found.")

    X, y_main, y_life = number_frequency_features(df, main_cols, life_col)

    # === Main numbers: Predict via probability scoring ===
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X, y_main)
    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    # Score and combine all candidates
    number_scores = {}
    for i in range(5):
        for idx, prob in enumerate(probas[i][0]):
            number = idx + 1
            number_scores[number] = number_scores.get(number, 0) + prob

    # Pick top 5 unique numbers
    top_5 = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    main_preds = [int(num) for num, _ in top_5]

    # === Life Ball ===
    life_model = RandomForestClassifier(n_estimators=100, random_state=42)
    life_model.fit(X, y_life)
    life_pred = int(life_model.predict(last_features)[0])

    return {
        "main_numbers": sorted(main_preds),
        "life_ball": life_pred
    }
