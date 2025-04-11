import pandas as pd
from collections import Counter
import os
import re

def analyze_setforlife_draws(return_raw=False):
    path = "data/setforlife_draws.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Make sure it's created before analysis.")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    main_cols = [col for col in df.columns if "ball" in col and "life" not in col]
    life_col = [col for col in df.columns if "life" in col][0]

    # Clean main numbers
    main_numbers_raw = df[main_cols].values.flatten()
    main_numbers = [int(re.search(r"\d+", str(val)).group()) for val in main_numbers_raw if pd.notna(val)]

    # Clean life balls
    life_balls = [int(re.search(r"\d+", str(val)).group()) for val in df[life_col] if pd.notna(val)]

    main_freq = pd.Series(main_numbers).value_counts().sort_values(ascending=False)
    life_freq = pd.Series(life_balls).value_counts().sort_values(ascending=False)

    print("🎯 Top Main Numbers:\n", main_freq.head(10))
    print("💜 Top Life Balls:\n", life_freq.head(5))

    if return_raw:
        return df, main_cols, life_col

    return {
        "main_number_weights": main_freq.to_dict(),
        "life_ball_weights": life_freq.to_dict()
    }
