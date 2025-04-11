# src/euromillions/euromillions_fetcher.py

import requests
import pandas as pd
import os

def fetch_draws():
    url = "https://euromillions.api.pedromealha.dev/v1/draws"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
        draws = json_data if isinstance(json_data, list) else json_data.get("items", [])
        records = []

        for draw in draws:
            try:
                main = [int(n) for n in draw.get("numbers", [])]
                stars = [int(s) for s in draw.get("stars", [])]
                date = draw.get("date")

                if len(main) != 5 or len(stars) != 2 or not date:
                    continue  # skip malformed entries

                records.append({
                    "date": date,
                    **{f"main_{i+1}": n for i, n in enumerate(main)},
                    **{f"star_{i+1}": s for i, s in enumerate(stars)}
                })
            except Exception as e:
                print(f"Skipping invalid draw: {e}")
                continue

        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv("data/euromillions_draws.csv", index=False)
        print(f"✅ Saved {len(df)} draws to data/euromillions_draws.csv")
    else:
        print(f"❌ API Error: {response.status_code}")
