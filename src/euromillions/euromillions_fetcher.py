import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

def fetch_euromillions_data():
    print("📥 Fetching EuroMillions data from official website...")
    url = "https://www.national-lottery.com/euromillions/results/history"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    # Extract and parse the table into a DataFrame (wrap with StringIO to avoid FutureWarning)
    df = pd.read_html(StringIO(str(table)))[0]

    # Rename columns to match analyzer expectations
    df.columns = [
        "draw_date", "ball_1", "ball_2", "ball_3", "ball_4", "ball_5",
        "lucky_star_1", "lucky_star_2", "jackpot", "draw_machine"
    ]

    print("✅ EuroMillions columns:", df.columns.tolist())
    return df


def save_euromillions_draws_csv():
    print("📥 Downloading EuroMillions data...")
    df = fetch_euromillions_data()
    df.to_csv("data/euromillions_draws.csv", index=False)
    print("✅ EuroMillions results saved to data/euromillions_draws.csv")
