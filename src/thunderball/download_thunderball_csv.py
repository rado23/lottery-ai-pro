import pandas as pd

# Thunderball CSV URL
url = "https://www.national-lottery.co.uk/results/thunderball/draw-history/csv"

# Load and clean
df = pd.read_csv(url)

# Normalize column names (optional but nice)
df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"), inplace=True)

# Preview what it looks like
print(df.head())

# Save to your local data folder
df.to_csv("data/thunderball_draws.csv", index=False)
print("✅ Thunderball results saved to data/thunderball_draws.csv")
