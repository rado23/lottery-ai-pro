import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    # === Train main number model ===
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    X_train, X_val, y_train, y_val = train_test_split(X, y_main, test_size=0.15, random_state=42)
    model.fit(X_train, y_train)

    y_pred_main = model.predict(X_val)
    acc_main = np.mean([
        accuracy_score(y_val.iloc[:, i], y_pred_main[:, i])
        for i in range(y_val.shape[1])
    ])
    print(f"✅ Main Number Model Accuracy: {acc_main:.3f}")

    # === Main number prediction with confidence ===
    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    number_scores = {}
    for i in range(5):
        for idx, prob in enumerate(probas[i][0]):
            number = idx + 1
            number_scores[number] = number_scores.get(number, 0) + prob

    sorted_candidates = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)

    main_preds = []
    seen = set()
    total_conf = 0
    for num, score in sorted_candidates:
        if num not in seen:
            seen.add(num)
            main_preds.append(num)
            total_conf += score
        if len(main_preds) == 5:
            break

    avg_main_conf = round(total_conf / 5, 4)

    # === Train life ball model ===
    life_model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train_life, X_val_life, y_train_life, y_val_life = train_test_split(
        X, y_life, test_size=0.15, random_state=42
    )
    life_model.fit(X_train_life, y_train_life)
    y_pred_life = life_model.predict(X_val_life)
    acc_life = accuracy_score(y_val_life, y_pred_life)
    print(f"✅ Life Ball Model Accuracy: {acc_life:.3f}")

    life_pred = int(life_model.predict(last_features)[0])
    life_conf = round(max(life_model.predict_proba(last_features)[0]), 4)

    return {
        "main_numbers": sorted(main_preds),
        "life_ball": life_pred,
        "confidence": {
            "main_numbers": avg_main_conf,
            "life_ball": life_conf
        }
    }


# Optional CLI test
if __name__ == "__main__":
    df = pd.read_csv("data/setforlife_draws.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if re.fullmatch(r"ball_[1-5]", col)]
    life_col = next((col for col in df.columns if "life" in col.lower()), None)

    result = predict_setforlife_with_ml(df, main_cols, life_col)
    print("🔮 Set for Life ML Prediction:")
    print("Main Numbers:", result["main_numbers"])
    print("Life Ball:", result["life_ball"])
    print("Confidence:", result["confidence"])
