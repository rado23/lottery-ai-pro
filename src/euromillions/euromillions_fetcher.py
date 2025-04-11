import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_euromillions_data():
    print("📥 Fetching EuroMillions data from official website...")
    url = "https://www.national-lottery.com/euromillions/results/history"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    # Extract and parse the table into a DataFrame
    df = pd.read_html(str(table))[0]

    # ✅ Normalize column names to ensure 'date' exists
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    print("✅ EuroMillions columns:", df.columns.tolist())  # Optional: log columns
    return df


def save_euromillions_draws_csv():
    print("📥 Downloading EuroMillions data...")
    df = fetch_euromillions_data()
    df.to_csv("data/euromillions_draws.csv", index=False)
    print("✅ EuroMillions results saved to data/euromillions_draws.csv")
