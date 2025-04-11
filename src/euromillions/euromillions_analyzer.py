import pandas as pd
from datetime import datetime
from collections import Counter
from itertools import combinations
import re

def clean_date_string(date_str):
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', str(date_str))

def analyze_euromillions_draws():
    df = pd.read_csv("data/euromillions_draws.csv")

    # Clean and convert date column
    df["draw_date"] = df["draw_date"].apply(clean_date_string)
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce", dayfirst=True)

    main_counts = Counter()
    star_counts = Counter()

    today = pd.Timestamp(datetime.today())

    for _, row in df.iterrows():
        date = row["draw_date"]
        weeks_ago = (today - date).days // 7

        main_numbers = [row[f"ball_{i}"] for i in range(1, 6)]
        star_numbers = [row[f"lucky_star_{i}"] for i in range(1, 3)]

        for num in main_numbers:
            main_counts[num] += max(1, 10 - weeks_ago)

        for star in star_numbers:
            star_counts[star] += max(1, 10 - weeks_ago)

    top_main = dict(main_counts.most_common(10))
    top_stars = dict(star_counts.most_common(5))

    print("🎯 Top Main Numbers:")
    print(pd.Series(top_main))

    print("⭐ Top Lucky Stars:")
    print(pd.Series(top_stars))

    return {
        "top_main_numbers": list(top_main.keys()),
        "top_star_numbers": list(top_stars.keys()),
    }
