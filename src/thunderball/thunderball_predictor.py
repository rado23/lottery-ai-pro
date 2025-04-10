import pandas as pd
import random
from collections import defaultdict

def weighted_random_sample(weight_dict, k):
    """
    Sample k unique numbers from a weighted dictionary.
    """
    items = list(weight_dict.items())
    population = [item[0] for item in items]
    weights = [item[1] for item in items]
    return random.choices(population=population, weights=weights, k=k)

def generate_thunderball_predictions(n_sets=10):
    # Load frequency data
    main_freq_path = "data/thunderball_main_frequency.csv"
    thunder_freq_path = "data/thunderball_star_frequency.csv"

    main_df = pd.read_csv(main_freq_path, index_col=0)
    thunder_df = pd.read_csv(thunder_freq_path, index_col=0)

    # Convert frequency DataFrames to weight dictionaries
    main_weights = defaultdict(int, main_df.to_dict()["count"])
    thunder_weights = defaultdict(int, thunder_df.to_dict()["count"])

    predictions = []

    for _ in range(n_sets):
        main_numbers = set()
        while len(main_numbers) < 5:
            sample = weighted_random_sample(main_weights, 1)[0]
            main_numbers.add(sample)

        thunderball = weighted_random_sample(thunder_weights, 1)[0]

        predictions.append({
            "main": sorted(main_numbers),
            "thunderball": thunderball
        })

    return predictions

# Debug mode: run directly
if __name__ == "__main__":
    sets = generate_thunderball_predictions()
    for i, p in enumerate(sets, 1):
        print(f"{i}: Main: {p['main']} | Thunderball: {p['thunderball']}")
