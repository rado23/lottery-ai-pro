# src/predictor.py

import random

def weighted_random_sample(weights_dict, k):
    numbers = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return random.choices(numbers, weights=weights, k=k)

def generate_predictions(stats, sets=10):
    predictions = []

    for _ in range(sets):
        main_numbers = set()
        while len(main_numbers) < 5:
            main_numbers.update(weighted_random_sample(stats["main_number_weights"], 5 - len(main_numbers)))
        main_numbers = sorted(main_numbers)

        lucky_stars = set()
        while len(lucky_stars) < 2:
            lucky_stars.update(weighted_random_sample(stats["star_number_weights"], 2 - len(lucky_stars)))
        lucky_stars = sorted(lucky_stars)

        predictions.append({
            "main_numbers": main_numbers,
            "lucky_stars": lucky_stars
        })

    return predictions
