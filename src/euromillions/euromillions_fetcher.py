import requests
import pandas as pd
import os

def fetch_euromillions_data():
    url = "https://api.national-lottery.co.uk/api/DrawHistory/EURO"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")

    data = response.json()

    all_draws = []
    for draw in data.get("drawHistory", []):
        row = {
            "date": draw.get("drawDate"),
            "main1": draw.get("balls", [None]*5)[0],
            "main2": draw.get("balls", [None]*5)[1],
            "main3": draw.get("balls", [None]*5)[2],
            "main4": draw.get("balls", [None]*5)[3],
            "main5": draw.get("balls", [None]*5)[4],
            "star1": draw.get("luckyStars", [None]*2)[0],
            "star2": draw.get("luckyStars", [None]*2)[1],
        }
        all_draws.append(row)

    df = pd.DataFrame(all_draws)
    return df


def save_euromillions_draws_csv(path="data/euromillions_draws.csv"):
    print("📥 Fetching EuroMillions data from API...")
    df = fetch_euromillions_data()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ EuroMillions data saved to {path}")
