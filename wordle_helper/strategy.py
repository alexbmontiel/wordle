"""
DEPRECATED: This module is deprecated. Please use the new modular structure:

- wordle_helper.core.result.compute_result
- wordle_helper.scoring.strategies.InformationGainScorer
- wordle_helper.game.player.Player
- wordle_helper.evaluation.simulator.simulate_game
- wordle_helper.evaluation.benchmark.evaluate_strategy

This module is kept for backward compatibility only.
"""

# Backward compatibility imports
from wordle_helper.scoring.strategies import InformationGainScorer

# For backward compatibility, export information_gain as a function
def information_gain(guess: str, word_list: dict) -> float:
    """DEPRECATED: Use InformationGainScorer instead."""
    scorer = InformationGainScorer()
    return scorer.score(guess, word_list)


def best_guess(word_list: dict[str, tuple[float, frozenset[str]]]) -> str | None:
    """DEPRECATED: Use Player.choose_guess() instead."""
    from wordle_helper.game.player import Player
    from wordle_helper.game.state import GameState
    
    scorer = InformationGainScorer()
    player = Player(scorer=scorer)
    state = GameState(remaining_words=word_list)
    return player.choose_guess(state)


def expected_remaining(guess: str, word_list: dict[str, tuple[float, frozenset[str]]]) -> float:
    """DEPRECATED: Use ExpectedRemainingScorer instead."""
    from wordle_helper.scoring.strategies import ExpectedRemainingScorer
    scorer = ExpectedRemainingScorer()
    return scorer.score(guess, word_list)


def rank_guesses(word_list: dict[str, tuple[float, frozenset[str]]], candidates: list[str] | None = None) -> list[tuple[str, float]]:
    """DEPRECATED: Use Scorer.rank_candidates() instead."""
    scorer = InformationGainScorer()
    return scorer.rank_candidates(word_list, candidates=candidates)


def play_game(
    answer: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
    starting_word: str,
    max_guesses: int = 6,
) -> int:
    """DEPRECATED: Use evaluation.simulator.simulate_game() instead."""
    from wordle_helper.game.player import Player
    from wordle_helper.evaluation.simulator import simulate_game
    
    scorer = InformationGainScorer()
    player = Player(scorer=scorer)
    
    result = simulate_game(
        answer=answer,
        player=player,
        word_list=word_list,
        starting_word=starting_word,
        max_guesses=max_guesses,
    )
    return result.guesses


def evaluate_starting_word(
    starting_word: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    show_progress: bool = True,
) -> dict:
    """DEPRECATED: Use evaluation.benchmark.evaluate_strategy() instead."""
    from wordle_helper.evaluation.benchmark import evaluate_strategy
    
    scorer = InformationGainScorer()
    result = evaluate_strategy(
        scorer=scorer,
        word_list=word_list,
        answers=answers,
        starting_word=starting_word,
        max_guesses=max_guesses,
        fail_penalty=fail_penalty,
        show_progress=show_progress,
    )
    
    # Convert EvaluationResult to dict for backward compatibility
    return {
        "starting_word": result.starting_word,
        "average_score": result.average_score,
        "failure_rate": result.failure_rate,
        "distribution": result.distribution,
        "total_games": result.total_games,
    }


def rank_starting_words(
    word_list: dict[str, tuple[float, frozenset[str]]],
    candidates: list[str] | None = None,
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
) -> list[dict]:
    """DEPRECATED: Use evaluation.benchmark with multiple starting words."""
    from wordle_helper.evaluation.benchmark import evaluate_strategy
    
    if candidates is None:
        candidates = list(word_list.keys())
    
    scorer = InformationGainScorer()
    results = []
    for starting_word in candidates:
        result = evaluate_strategy(
            scorer=scorer,
            word_list=word_list,
            answers=answers,
            starting_word=starting_word,
            max_guesses=max_guesses,
            fail_penalty=fail_penalty,
            show_progress=False,
        )
        results.append({
            "starting_word": result.starting_word,
            "average_score": result.average_score,
            "failure_rate": result.failure_rate,
            "distribution": result.distribution,
            "total_games": result.total_games,
        })
    
    results.sort(key=lambda x: x["average_score"])
    return results


def rank_starting_words_parallel(
    word_list: dict[str, tuple[float, frozenset[str]]],
    candidates: list[str] | None = None,
    answers: list[str] | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    n_workers: int | None = None,
) -> list[dict]:
    """DEPRECATED: Use FastEvaluator from fast_strategy or evaluation.optimized_evaluator instead."""
    # For now, fall back to sequential version
    # TODO: Implement parallel version using FastEvaluator
    return rank_starting_words(
        word_list=word_list,
        candidates=candidates,
        answers=answers,
        max_guesses=max_guesses,
        fail_penalty=fail_penalty,
    )
