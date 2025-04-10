import pandas as pd
from collections import Counter
import re

# Load and normalize
df = pd.read_csv("data/thunderball_draws.csv")
df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"), inplace=True)

# Find columns
main_cols = [col for col in df.columns if "ball" in col and "thunder" not in col]
thunder_col = [col for col in df.columns if "thunder" in col][0]

# Clean and flatten main numbers
main_numbers_raw = df[main_cols].values.flatten()
main_numbers = []

for value in main_numbers_raw:
    match = re.search(r"\d+", str(value))  # Extract number part
    if match:
        main_numbers.append(int(match.group()))

# Clean Thunderball column
thunderballs = []
for value in df[thunder_col]:
    match = re.search(r"\d+", str(value))
    if match:
        thunderballs.append(int(match.group()))

# Frequency counts
main_freq = pd.Series(main_numbers).value_counts().sort_values(ascending=False)
thunder_freq = pd.Series(thunderballs).value_counts().sort_values(ascending=False)

# Output
print("🎯 Top Main Numbers:\n", main_freq.head(10))
print("\n⭐ Top Thunderballs:\n", thunder_freq.head(5))

# Optional: save for modeling
main_freq.to_csv("data/thunderball_main_frequency.csv")
thunder_freq.to_csv("data/thunderball_star_frequency.csv")

def analyze_thunderball_draws():
    # your existing code...
    return df, main_cols, thunder_col

# ✅ Required for app.py to import it
if __name__ == "__main__":
    analyze_thunderball_draws()
