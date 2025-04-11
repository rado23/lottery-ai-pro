
import pandas as pd
import random
from src.thunderball.thunderball_analyzer import analyze_thunderball_draws

def generate_thunderball_predictions(stats, num_predictions=10):
    main_freq = stats["main_freq"]
    thunder_freq = stats["thunder_freq"]

    main_numbers = list(main_freq.keys())
    thunder_numbers = list(thunder_freq.keys())

    main_probs = [main_freq[n] for n in main_numbers]
    thunder_probs = [thunder_freq[t] for t in thunder_numbers]

    main_probs = [p / sum(main_probs) for p in main_probs]
    thunder_probs = [p / sum(thunder_probs) for p in thunder_probs]

    predictions = []
    for _ in range(num_predictions * 2):
        main = sorted(random.choices(main_numbers, weights=main_probs, k=5))
        thunder = random.choices(thunder_numbers, weights=thunder_probs, k=1)
        predictions.append((main, thunder))

    unique = {}
    for main, thunder in predictions:
        key = tuple(main + thunder)
        score = sum(main_freq[n] for n in main) + thunder_freq[thunder[0]]
        unique[key] = score

    ranked = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    top_10 = [list(k[:5]) + list(k[5:]) for k, _ in ranked[:10]]
    return [{"main_numbers": combo[:5], "thunderball": combo[5]} for combo in top_10]
