# src/euromillions_predictor.py
import pandas as pd
import numpy as np

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
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws

def generate_euromillions_predictions(stats, num_predictions=10):
    main_weights = stats["main_number_weights"]
    star_weights = stats["star_number_weights"]

    main_numbers = list(main_weights.keys())
    star_numbers = list(star_weights.keys())

    main_probs = [main_weights[n] for n in main_numbers]
    star_probs = [star_weights[s] for s in star_numbers]

    # Normalize to probabilities
    main_probs = [p / sum(main_probs) for p in main_probs]
    star_probs = [p / sum(star_probs) for p in star_probs]

    predictions = []
    for _ in range(num_predictions * 2):  # generate more for ranking
        main = weighted_unique_sample(main_numbers, main_probs, 5)
        stars = weighted_unique_sample(star_numbers, star_probs, 2)
        predictions.append((main, stars))

    # De-duplicate and rank by total weight (most probable first)
    unique = {}
    for main, stars in predictions:
        key = tuple(main + stars)
        score = sum(main_weights[n] for n in main) + sum(star_weights[s] for s in stars)
        unique[key] = score

    ranked = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    top_10 = [list(k[:5]) + list(k[5:]) for k, _ in ranked[:10]]
    return [{"main_numbers": combo[:5], "lucky_stars": combo[5:]} for combo in top_10]
