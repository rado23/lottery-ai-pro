# src/thunderball/thunderball_analyzer.py

import pandas as pd
from datetime import datetime
from collections import Counter
import re
import os

def analyze_thunderball_draws():
    path = "data/thunderball_draws.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure it's created before analysis.")

    # Load and normalize
    df = pd.read_csv(path)
    df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"), inplace=True)

    # Find columns
    main_cols = [col for col in df.columns if "ball" in col and "thunder" not in col]
    thunder_col = [col for col in df.columns if "thunder" in col][0]

    # Clean and flatten main numbers
    main_numbers_raw = df[main_cols].values.flatten()
    main_numbers = []
    for value in main_numbers_raw:
        match = re.search(r"\d+", str(value))
        if match:
            main_numbers.append(int(match.group()))

    # Clean Thunderball column
    thunderballs = []
    for value in df[thunder_col]:
        match = re.search(r"\d+", str(value))
        if match:
            thunderballs.append(int(match.group()))

    # Frequency counts
    main_freq = pd.Series(main_numbers).value_counts().sort_values(ascending=False)
    thunder_freq = pd.Series(thunderballs).value_counts().sort_values(ascending=False)

    # Optional outputs (for logs or saving to disk)
    print("🎯 Top Main Numbers:\n", main_freq.head(10))
    print("\n⭐ Top Thunderballs:\n", thunder_freq.head(5))

    # Save for modeling
    main_freq.to_csv("data/thunderball_main_frequency.csv")
    thunder_freq.to_csv("data/thunderball_star_frequency.csv")

    return {
        "main_number_weights": main_freq.to_dict(),
        "thunderball_weights": thunder_freq.to_dict()
    }

# ✅ Optional manual run
if __name__ == "__main__":
    analyze_thunderball_draws()
