# src/strategy/strategy_planner.py

from src.config.game_meta import GAME_META

def build_strategy(funds: float, selected_games: list[str] = None, preference: str = "any_big_win", max_draws: int = 1):
    """
    Builds a strategy by distributing funds across games to optimize for the selected preference.

    Args:
        funds (float): total user budget
        selected_games (list[str]): list of game names (or None = all)
        preference (str): 'any_win', 'any_big_win', 'highest_win'
        max_draws (int): number of draws to plan for

    Returns:
        dict with game-wise ticket allocations and expected value (simplified estimate)
    """
    if selected_games is None:
        selected_games = list(GAME_META.keys())

    strategy = {}
    total_score = 0

    for game in selected_games:
        meta = GAME_META[game]
        price = meta["ticket_price"]
        num_draws = min(max_draws, meta["draws_per_week"])

        max_tickets = int(funds // (price * num_draws))
        if max_tickets == 0:
            continue

        tickets = max_tickets
        funds -= tickets * price * num_draws

        prize_weights = list(meta["prize_levels"].values())
        if preference == "any_win":
            score = sum(prize_weights)
        elif preference == "any_big_win":
            score = max(prize_weights)
        elif preference == "highest_win":
            score = meta["prize_levels"]["jackpot"]
        else:
            score = sum(prize_weights)

        weighted_score = tickets * score
        total_score += weighted_score

        strategy[game] = {
            "tickets_per_draw": tickets,
            "ticket_price": price,
            "total_spent": tickets * price * num_draws,
            "draws": num_draws,
            "priority_score": weighted_score,
            "preference": preference,
        }

    strategy["meta"] = {
        "funds_remaining": round(funds, 2),
        "preference": preference,
        "draws_considered": max_draws,
        "score_total": total_score
    }

    return strategy
