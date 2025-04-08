# main.py

from src.fetcher import fetch_draws
from src.analyzer import analyze_draws
from src.predictor import generate_predictions
from src.analyzer import analyze_co_occurrences
from src.ml_predictor import generate_ml_predictions

def main():
    print("Fetching EuroMillions draw data...")
    fetch_draws()

    print("Analyzing draw data...")
    stats = analyze_draws()

    print("Generating predictions...")
    predictions = generate_predictions(stats)

    print("\nTop 10 predicted number sets:")
    for i, p in enumerate(predictions, 1):
        print(f"{i}: Main: {p['main_numbers']}, Stars: {p['lucky_stars']}")

    print("\nAnalyzing co-occurring number pairs...")
    co_occurrences = analyze_co_occurrences()

    print("\nTop 10 Main Number Pairs:")
    for pair, count in co_occurrences["main_number_pairs"][:10]:
        print(f"{pair}: {count} times")

    print("\nTop Lucky Star Pairs:")
    for pair, count in co_occurrences["star_number_pairs"][:5]:
        print(f"{pair}: {count} times")

    print("\n🎯 ML-Based Prediction:")
    ml_prediction = generate_ml_predictions()
    print(f"Main Numbers: {ml_prediction['main_numbers']}")
    print(f"Lucky Stars: {ml_prediction['lucky_stars']}")


if __name__ == "__main__":
    main()
