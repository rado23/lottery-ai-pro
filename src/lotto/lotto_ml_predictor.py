import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def number_frequency_features(df, main_cols):
    mlb = MultiLabelBinarizer(classes=range(1, 60))  # Lotto numbers: 1–59
    one_hot = mlb.fit_transform(df[main_cols].values.tolist())
    X = pd.DataFrame(one_hot[:-1])
    y = pd.DataFrame(one_hot[1:], columns=[f"n_{i+1}" for i in range(59)])
    return X, y


def predict_lotto_with_ml(df, main_cols):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if "ball" in col]

    X, y = number_frequency_features(df, main_cols)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"✅ Lotto Model Accuracy: {acc:.3f}")

    # Predict on latest draw
    last_features = X.iloc[[-1]]
    probas = model.predict_proba(last_features)

    number_scores = {}
    for idx, prob in enumerate(probas):
        if len(prob[0]) == 2:
            number_scores[idx + 1] = prob[0][1]  # probability of being drawn
        else:
            number_scores[idx + 1] = 0  # fallback

    sorted_probs = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)

    selected_numbers = []
    seen = set()
    confidence_sum = 0

    for num, score in sorted_probs:
        if num not in seen:
            seen.add(num)
            selected_numbers.append(num)
            confidence_sum += score
        if len(selected_numbers) == 6:
            break

    avg_confidence = round(confidence_sum / 6, 4)

    return {
        "main_numbers": sorted(selected_numbers),
        "confidence": avg_confidence
    }


# Optional CLI test
if __name__ == "__main__":
    df = pd.read_csv("data/lotto_draws.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    main_cols = [col for col in df.columns if "ball" in col]
    result = predict_lotto_with_ml(df, main_cols)
    print("🔮 Lotto ML Prediction:")
    print("Main Numbers:", result["main_numbers"])
    print("Confidence:", result["confidence"])
