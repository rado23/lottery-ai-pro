# main.py

from src.euromillions.euromillions_fetcher import fetch_draws
from src.euromillions.euromillions_analyzer import analyze_euromillions_draws
from src.euromillions.euromillions_predictor import generate_euromillions_predictions
from src.euromillions.euromillions_analyzer import analyze_co_occurrences
from src.euromillions.euromillions_ml_predictor import predict_euromillions_with_ml

def main():
    print("Fetching EuroMillions draw data...")
    fetch_draws()

    print("Analyzing draw data...")
    stats = analyze_euromillions_draws()

    print("Generating predictions...")
    predictions = generate_euromillions_predictions(stats)

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
    ml_prediction = predict_euromillions_with_ml()
    print(f"Main Numbers: {ml_prediction['main_numbers']}")
    print(f"Lucky Stars: {ml_prediction['lucky_stars']}")


if __name__ == "__main__":
    main()
