# src/lotto/lotto_analyzer.py

import pandas as pd
from collections import Counter
import re
import os

def analyze_lotto_draws(return_raw=False):
    path = "data/lotto_draws.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure it's created before analysis.")

    df = pd.read_csv(path)
    df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"), inplace=True)

    # Identify main number columns
    main_cols = [col for col in df.columns if "ball" in col]

    # Flatten and clean main number values
    main_numbers_raw = df[main_cols].values.flatten()
    main_numbers = []
    for val in main_numbers_raw:
        match = re.search(r"\d+", str(val))
        if match:
            main_numbers.append(int(match.group()))

    # Count frequency
    main_freq = pd.Series(main_numbers).value_counts().sort_values(ascending=False)

    # Log results
    print("🎯 Top Main Numbers:\n", main_freq.head(10))

    # Save for modeling
    main_freq.to_csv("data/lotto_main_frequency.csv")

    if return_raw:
        return df, main_cols
    else:
        return {
            "main_number_weights": main_freq.to_dict()
        }

# Optional manual run
if __name__ == "__main__":
    analyze_lotto_draws()
