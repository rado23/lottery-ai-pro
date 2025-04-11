# src/lotto/download_lotto_csv.py

import pandas as pd
import os

def save_lotto_draws_csv():
    print("📥 Downloading Lotto data...")

    url = "https://www.national-lottery.co.uk/results/lotto/draw-history/csv"
    df = pd.read_csv(url)

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Save the data locally
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/lotto_draws.csv", index=False)

    print("✅ Lotto results saved to data/lotto_draws.csv")
