# src/euromillions_predictor.py

import pandas as pd
import numpy as np
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
import random

def weighted_unique_sample(population, weights, k):
    """
    Sample k unique elements from a weighted population without replacement.
    """
    population = list(population)
    weights = np.array(weights)
    selected = []
    for _ in range(k):
        total = weights.sum()
        probs = weights / total
        choice = np.random.choice(population, p=probs)
        idx = population.index(choice)
        selected.append(choice)
        population.pop(idx)
        weights = np.delete(weights, idx)
    return sorted(selected)

def generate_prediction(stats, k=10):
    all_main = stats["main_counts"].sort_values(ascending=False).index.tolist()
    all_stars = stats["star_counts"].sort_values(ascending=False).index.tolist()

    predictions = []
    used_main_sets = set()

    attempts = 0
    while len(predictions) < k and attempts < 1000:
        mains = sorted(random.sample(all_main[:25], 5))
        stars = sorted(random.sample(all_stars[:8], 2))
        main_key = tuple(mains)

        if main_key not in used_main_sets:
            used_main_sets.add(main_key)
            predictions.append({
                "main_numbers": [int(x) for x in mains],
                "lucky_stars": [int(x) for x in stars],
            })

        attempts += 1

    return predictions

def predict_euromillions():
    stats = analyze_euromillions_draws()
    predictions = generate_prediction(stats, k=10)

    return {"heuristic": predictions}
