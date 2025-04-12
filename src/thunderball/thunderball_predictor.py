import pandas as pd
import random

def weighted_unique_sample(choices, weights, k):
    """Sample k unique items based on weights without replacement."""
    pool = list(zip(choices, weights))
    result = []
    for _ in range(min(k, len(pool))):
        total_weight = sum(w for _, w in pool)
        probs = [w / total_weight for _, w in pool]
        pick = random.choices(pool, weights=probs, k=1)[0][0]
        result.append(pick)
        pool = [item for item in pool if item[0] != pick]
    return sorted(result)

def generate_thunderball_predictions(stats, num_predictions=10):
    df, main_cols, thunder_col = stats["df"], stats["main_cols"], stats["thunder_col"]

    # Clean and flatten main numbers
    main_numbers = pd.to_numeric(df[main_cols].values.flatten(), errors="coerce")
    main_numbers = [int(n) for n in main_numbers if pd.notna(n)]

    thunder_numbers = pd.to_numeric(df[thunder_col], errors="coerce")
    thunder_numbers = [int(t) for t in thunder_numbers if pd.notna(t)]

    # Frequency
    main_freq = pd.Series(main_numbers).value_counts().sort_values(ascending=False)
    thunder_freq = pd.Series(thunder_numbers).value_counts().sort_values(ascending=False)

    main_probs = [main_freq[n] for n in main_freq.index]
    thunder_probs = [thunder_freq[t] for t in thunder_freq.index]

    main_probs = [p / sum(main_probs) for p in main_probs]
    thunder_probs = [p / sum(thunder_probs) for p in thunder_probs]

    predictions = []
    for _ in range(num_predictions * 2):  # extra for deduplication
        main = weighted_unique_sample(list(main_freq.index), main_probs, k=5)
        thunder = random.choices(list(thunder_freq.index), weights=thunder_probs, k=1)
        predictions.append((main, thunder))

    # Deduplicate and rank by score
    unique = {}
    for main, thunder in predictions:
        key = tuple(main + thunder)
        score = sum(main_freq[n] for n in main) + thunder_freq[thunder[0]]
        unique[key] = score

    ranked = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    top_10 = [list(k[:5]) + list(k[5:]) for k, _ in ranked[:10]]
    return [{"main_numbers": combo[:5], "thunderball": combo[5]} for combo in top_10]
