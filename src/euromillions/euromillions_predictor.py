
import pandas as pd
import random
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
        main = sorted(random.choices(main_numbers, weights=main_probs, k=5))
        stars = sorted(random.choices(star_numbers, weights=star_probs, k=2))
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
