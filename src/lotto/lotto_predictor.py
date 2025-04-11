# src/lotto/lotto_predictor.py

import pandas as pd
import random
from src.lotto.lotto_analyzer import analyze_lotto_draws

def generate_lotto_predictions(stats, num_predictions=10):
    main_freq = stats["main_number_weights"]
    main_numbers = list(main_freq.keys())
    main_probs = [main_freq[n] for n in main_numbers]
    main_probs = [p / sum(main_probs) for p in main_probs]

    predictions = []
    for _ in range(num_predictions * 2):  # allow for duplicates to rank later
        selected = sorted(random.choices(main_numbers, weights=main_probs, k=6))
        score = sum(main_freq[n] for n in selected)
        predictions.append((tuple(selected), score))

    # Deduplicate and rank
    unique = {}
    for numbers, score in predictions:
        unique[numbers] = score

    ranked = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    return [list(k) for k, _ in ranked[:10]]
