# src/euromillions/euromillions_analyzer.py

import pandas as pd
from collections import Counter
from datetime import datetime
from itertools import combinations
from collections import defaultdict

def analyze_euromillions_draws():
    df = pd.read_csv("data/euromillions_draws.csv")

    main_cols = [f"main_{i+1}" for i in range(5)]
    star_cols = [f"star_{i+1}" for i in range(2)]

    all_main_numbers = df[main_cols].values.flatten()
    all_star_numbers = df[star_cols].values.flatten()

    main_counts = pd.Series(all_main_numbers).value_counts().sort_values(ascending=False)
    star_counts = pd.Series(all_star_numbers).value_counts().sort_values(ascending=False)

    return {
        "main_counts": main_counts,
        "star_counts": star_counts,
        "df": df,
        "main_cols": main_cols,
        "star_cols": star_cols
    }


def analyze_co_occurrences(filepath="data/euromillions_draws.csv"):
    df = pd.read_csv(filepath)

    main_pair_counts = defaultdict(int)
    star_pair_counts = defaultdict(int)

    for _, row in df.iterrows():
        main_nums = [row[f"main_{i+1}"] for i in range(5)]
        star_nums = [row[f"star_{i+1}"] for i in range(2)]

        # Count all main number pairs (unordered)
        for pair in combinations(sorted(main_nums), 2):
            main_pair_counts[pair] += 1

        # Count all lucky star pairs
        star_pair = tuple(sorted(star_nums))
        star_pair_counts[star_pair] += 1

    # Sort by most common
    main_top = sorted(main_pair_counts.items(), key=lambda x: x[1], reverse=True)
    star_top = sorted(star_pair_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "main_number_pairs": main_top,
        "star_number_pairs": star_top
    }
