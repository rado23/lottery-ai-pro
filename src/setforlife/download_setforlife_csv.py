import pandas as pd
import os

def save_setforlife_draws_csv():
    url = "https://www.national-lottery.co.uk/results/set-for-life/draw-history/csv"
    df = pd.read_csv(url)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Save the file
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/setforlife_draws.csv", index=False)
    print("✅ Set for Life data saved to data/setforlife_draws.csv")
