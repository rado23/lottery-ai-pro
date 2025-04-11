import requests
import pandas as pd
from datetime import datetime
from io import StringIO

EUROMILLIONS_CSV_URL = "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv"
SAVE_PATH = "data/euromillions_draws.csv"

def fetch_euromillions_data():
    print("📥 Fetching EuroMillions data from official website...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.114 Safari/537.36"
        )
    }

    response = requests.get(EUROMILLIONS_CSV_URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    csv_data = response.content.decode("utf-8")
    df = pd.read_csv(StringIO(csv_data))

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df

def save_euromillions_draws_csv():
    df = fetch_euromillions_data()
    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ EuroMillions results saved to {SAVE_PATH}")
