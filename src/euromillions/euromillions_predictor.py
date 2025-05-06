# src/euromillions/euromillions_predictor.py

import random
import pandas as pd

def analyze_euromillions_draws():
    df = pd.read_csv("data/euromillions_draws.csv")

    # Main number frequency
    main_numbers = df[[f"main_{i+1}" for i in range(5)]].values.ravel()
    main_counts = pd.Series(main_numbers).value_counts()

    # Star number frequency
    stars = df[[f"star_{i+1}" for i in range(2)]].values.ravel()
    star_counts = pd.Series(stars).value_counts()

    return {
        "main_counts": main_counts,
        "star_counts": star_counts
    }

def generate_euromillions_predictions(stats, total_sets=10):
    predictions = []

    all_main = stats["main_counts"].sort_values(ascending=False).index.tolist()
    all_stars = stats["star_counts"].sort_values(ascending=False).index.tolist()

    while len(predictions) < total_sets:
        # Select top 25% most frequent + some randomness
        top_main = all_main[:25]
        top_stars = all_stars[:10]

        main = sorted(random.sample(top_main + random.sample(all_main, 20), 5))
        stars = sorted(random.sample(top_stars + random.sample(all_stars, 5), 2))

        # Ensure uniqueness
        if len(set(main)) == 5 and len(set(stars)) == 2:
            predictions.append({
                "main_numbers": [int(n) for n in main],
                "lucky_stars": [int(s) for s in stars]
            })

    return predictions
