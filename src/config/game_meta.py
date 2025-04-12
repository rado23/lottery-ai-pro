# src/strategy/game_meta.py

GAME_META = {
    "euromillions": {
        "ticket_price": 2.5,
        "prize_levels": {
            "jackpot": 50_000_000,
            "5+1": 100_000,
            "5": 10_000,
            "4+2": 500,
            "4+1": 100,
            "4": 50,
            "3+2": 25,
            "3+1": 15,
            "2+2": 10,
            "3": 8,
            "1+2": 5,
            "2+1": 4
        },
        "draws_per_week": 2
    },
    "thunderball": {
        "ticket_price": 1.0,
        "prize_levels": {
            "jackpot": 500_000,
            "5+tb": 5_000,
            "5": 250,
            "4+tb": 100,
            "4": 20,
            "3+tb": 20,
            "3": 10,
            "2+tb": 10,
            "1+tb": 5,
            "0+tb": 3
        },
        "draws_per_week": 4
    },
    "lotto": {
        "ticket_price": 2.0,
        "prize_levels": {
            "jackpot": 2_000_000,
            "5": 1750,
            "4": 140,
            "3": 30,
            "2": 0
        },
        "draws_per_week": 2
    },
    "setforlife": {
        "ticket_price": 1.5,
        "prize_levels": {
            "jackpot": 3_600_000,  # £10,000 x 12 months x 30 years
            "5+0": 120_000,        # £10,000 x 12 months
            "4+1": 250,
            "4+0": 50,
            "3+1": 30,
            "3+0": 20,
            "2+1": 10,
            "2+0": 5
        },
        "draws_per_week": 2
    }
}
