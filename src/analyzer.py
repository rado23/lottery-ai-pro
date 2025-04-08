# src/analyzer.py

import pandas as pd
from collections import Counter
from datetime import datetime
from itertools import combinations
from collections import defaultdict

def analyze_draws():
    df = pd.read_csv("data/euromillions_draws.csv")
    df["date"] = pd.to_datetime(df["date"])

    main_counts = Counter()
    star_counts = Counter()

    today = pd.Timestamp(datetime.today())
    for _, row in df.iterrows():
        days_ago = (today - row["date"]).days
        weight = max(1, 100 - days_ago // 7)  # recency weight: stronger for recent weeks

        for i in range(1, 6):
            main_counts[row[f"main_{i}"]] += weight
        for i in range(1, 3):
            star_counts[row[f"star_{i}"]] += weight

    return {
        "main_number_weights": dict(main_counts),
        "star_number_weights": dict(star_counts)
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

