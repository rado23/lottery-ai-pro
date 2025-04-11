import random

def generate_setforlife_predictions(stats, num_predictions=10):
    main_freq = stats["main_number_weights"]
    life_freq = stats["life_ball_weights"]

    main_numbers = list(main_freq.keys())
    life_numbers = list(life_freq.keys())

    main_probs = [main_freq[n] for n in main_numbers]
    life_probs = [life_freq[t] for t in life_numbers]

    main_probs = [p / sum(main_probs) for p in main_probs]
    life_probs = [p / sum(life_probs) for p in life_probs]

    predictions = []
    for _ in range(num_predictions * 2):  # generate more for uniqueness filtering
        main = sorted(random.choices(main_numbers, weights=main_probs, k=5))
        life = random.choices(life_numbers, weights=life_probs, k=1)
        predictions.append((main, life))

    unique = {}
    for main, life in predictions:
        key = tuple(main + life)
        score = sum(main_freq[n] for n in main) + life_freq[life[0]]
        unique[key] = score

    ranked = sorted(unique.items(), key=lambda x: x[1], reverse=True)
    top_10 = [list(k[:5]) + list(k[5:]) for k, _ in ranked[:10]]

    return [{"main_numbers": combo[:5], "life_ball": combo[5]} for combo in top_10]
