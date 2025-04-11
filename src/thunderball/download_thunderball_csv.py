import pandas as pd

THUNDERBALL_CSV_URL = "https://www.national-lottery.co.uk/results/thunderball/draw-history/csv"
SAVE_PATH = "data/thunderball_draws.csv"

def save_thunderball_draws_csv():
    try:
        print("📥 Downloading Thunderball data...")
        df = pd.read_csv(THUNDERBALL_CSV_URL)
        df.to_csv(SAVE_PATH, index=False)
        print(f"✅ Thunderball data saved to {SAVE_PATH}")
    except Exception as e:
        print(f"❌ Error downloading or saving Thunderball data: {e}")
