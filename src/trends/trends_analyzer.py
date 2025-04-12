# src/trends/trends_analyzer.py

import pandas as pd
from collections import Counter
import os

def analyze_trends(game: str, draws: int = 100, short_term_window: int = 20):
    """
    Analyzes trends for a given lottery game.

    Args:
        game (str): One of 'euromillions', 'thunderball', 'lotto', 'setforlife'
        draws (int): Number of recent draws to analyze
        short_term_window (int): Number of draws to define short-term trends

    Returns:
        dict: Dictionary containing trends, frequencies, heatmaps, etc.
    """
    path = f"data/{game}_draws.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Draw data not found for {game}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    total_draws = len(df)
    df_recent = df.head(draws)
    df_short = df.head(short_term_window)

    # Determine main and secondary columns
    if game == "euromillions":
        main_cols = [col for col in df.columns if "main" in col]
        secondary_cols = [col for col in df.columns if "star" in col]
    elif game == "thunderball":
        main_cols = [col for col in df.columns if "ball" in col and "thunder" not in col]
        secondary_cols = [col for col in df.columns if "thunder" in col]
    elif game == "lotto":
        main_cols = [col for col in df.columns if "ball" in col]
        secondary_cols = []
    elif game == "setforlife":
        main_cols = [col for col in df.columns if "ball" in col and "life" not in col]
        secondary_cols = [col for col in df.columns if "life" in col]
    else:
        raise ValueError(f"Unsupported game: {game}")

    # Flatten helper
    def flatten(colset):
        return [int(str(x).replace("T", "").replace("SFL", "")) for x in df_recent[colset].values.flatten() if pd.notna(x)]

    # --- Frequency Counts ---
    main_values = flatten(main_cols)
    main_freq = pd.Series(main_values).value_counts().sort_values(ascending=False)

    secondary_freq = pd.Series()
    if secondary_cols:
        secondary_values = flatten(secondary_cols)
        secondary_freq = pd.Series(secondary_values).value_counts().sort_values(ascending=False)

    # --- Hot/Cold from Short-Term Window ---
    def flatten_short(colset):
        return [int(str(x).replace("T", "").replace("SFL", "")) for x in df_short[colset].values.flatten() if pd.notna(x)]

    short_main = flatten_short(main_cols)
    hot_main = pd.Series(short_main).value_counts().head(10).index.tolist()
    cold_main = pd.Series(short_main).value_counts().tail(10).index.tolist()

    hot_secondary = []
    cold_secondary = []
    if secondary_cols:
        short_sec = flatten_short(secondary_cols)
        freq_sec = pd.Series(short_sec).value_counts()
        hot_secondary = freq_sec.head(5).index.tolist()
        cold_secondary = freq_sec.tail(5).index.tolist()

    # --- Heatmap ---
    heatmap = {}
    all_heat_cols = main_cols + secondary_cols
    for col in all_heat_cols:
        values = [int(str(x).replace("T", "").replace("SFL", "")) for x in df_recent[col] if pd.notna(x)]
        counter = Counter(values)
        max_val = max(counter.values()) if counter else 1
        heatmap[col] = {k: round(v / max_val, 2) for k, v in counter.items()}  # Normalize 0–1

    # --- Interval: how long ago each number last appeared ---
    interval_map = {}
    all_numbers = flatten(main_cols + secondary_cols)
    appearance_tracker = {}

    for idx, row in df_recent.iterrows():
        for col in main_cols + secondary_cols:
            val = int(str(row[col]).replace("T", "").replace("SFL", "")) if pd.notna(row[col]) else None
            if val is not None and val not in appearance_tracker:
                appearance_tracker[val] = idx

    interval_map = {num: draws - idx for num, idx in appearance_tracker.items()}

    # Build response
    return {
        "metadata": {
            "game": game,
            "total_draws": total_draws,
            "sampled_draws": draws,
            "recent_window": short_term_window
        },
        "frequency": {
            "main_numbers": main_freq.to_dict(),
            "secondary_numbers": secondary_freq.to_dict() if not secondary_freq.empty else {}
        },
        "hot_numbers": {
            "main_numbers": hot_main,
            "secondary_numbers": hot_secondary
        },
        "cold_numbers": {
            "main_numbers": cold_main,
            "secondary_numbers": cold_secondary
        },
        "intervals": interval_map,
        "heatmap": heatmap
    }
