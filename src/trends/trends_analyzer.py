# src/trends/trends_analyzer.py

import pandas as pd
from collections import Counter
import os

def analyze_trends(game: str, draws: int = 100):
    path = f"data/{game}_draws.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Draw data not found for {game}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    df_recent = df.head(draws)
    trends = {}

    if game == "euromillions":
        main_cols = [col for col in df.columns if "main" in col]
        star_cols = [col for col in df.columns if "star" in col]

        main_values = df_recent[main_cols].values.flatten()
        main_values = [int(x) for x in main_values if pd.notna(x)]

        star_values = df_recent[star_cols].values.flatten()
        star_values = [int(x) for x in star_values if pd.notna(x)]

        main_freq = pd.Series(main_values).value_counts().sort_values(ascending=False)
        star_freq = pd.Series(star_values).value_counts().sort_values(ascending=False)

        trends["frequency"] = {
            "main_numbers": main_freq.to_dict(),
            "lucky_stars": star_freq.to_dict()
        }
        trends["hot_numbers"] = {
            "main_numbers": main_freq.head(10).index.tolist(),
            "lucky_stars": star_freq.head(5).index.tolist()
        }
        trends["cold_numbers"] = {
            "main_numbers": main_freq.tail(10).index.tolist(),
            "lucky_stars": star_freq.tail(5).index.tolist()
        }

        # Intervals for main numbers
        appearance_tracker = {}
        for idx, row in df_recent.iterrows():
            for col in main_cols:
                value = int(row[col])
                if value not in appearance_tracker:
                    appearance_tracker[value] = idx
        trends["intervals"] = {num: draws - idx for num, idx in appearance_tracker.items()}

        return trends

    elif game in ["thunderball", "setforlife"]:
        main_cols = [col for col in df.columns if "ball" in col and ("thunder" not in col and "life" not in col)]
        star_col = [col for col in df.columns if "thunder" in col or "life" in col][0]

    elif game == "lotto":
        main_cols = [col for col in df.columns if "ball" in col]
        star_col = None

    else:
        raise ValueError(f"Unsupported game: {game}")

    # Common logic for thunderball, lotto, setforlife
    main_values = df_recent[main_cols].values.flatten()
    main_values = [int(str(x).replace("T", "").replace("SFL", "")) for x in main_values if pd.notna(x)]
    main_freq = pd.Series(main_values).value_counts().sort_values(ascending=False)

    trends["frequency"] = {
        "main_numbers": main_freq.to_dict()
    }
    trends["hot_numbers"] = {
        "main_numbers": main_freq.head(10).index.tolist()
    }
    trends["cold_numbers"] = {
        "main_numbers": main_freq.tail(10).index.tolist()
    }

    if star_col:
        star_values = df_recent[star_col].apply(lambda x: int(str(x).replace("T", "").replace("SFL", "")))
        star_freq = star_values.value_counts().sort_values(ascending=False)
        trends["frequency"]["secondary_numbers"] = star_freq.to_dict()
        trends["hot_numbers"]["secondary_numbers"] = star_freq.head(5).index.tolist()
        trends["cold_numbers"]["secondary_numbers"] = star_freq.tail(5).index.tolist()

    # Intervals for main numbers
    appearance_tracker = {}
    for idx, row in df_recent.iterrows():
        for col in main_cols:
            value = int(str(row[col]).replace("T", "").replace("SFL", ""))
            if value not in appearance_tracker:
                appearance_tracker[value] = idx
    trends["intervals"] = {num: draws - idx for num, idx in appearance_tracker.items()}

    return trends
