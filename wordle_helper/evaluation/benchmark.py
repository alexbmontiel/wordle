"""Benchmarking and evaluation of strategies."""

from dataclasses import dataclass
from tqdm import tqdm
from wordle_helper.scoring.base import Scorer
from wordle_helper.game.player import Player
from wordle_helper.evaluation.simulator import simulate_game
from wordle_helper.data.word_list import WordList


@dataclass
class EvaluationResult:
    """
    Results from evaluating a strategy.
    
    Attributes:
        starting_word: The evaluated starting word
        average_score: Mean guesses (failures count as fail_penalty)
        failure_rate: Fraction of games that failed
        distribution: Dict mapping guess_count -> number of games
        total_games: Number of games played
    """
    starting_word: str
    average_score: float
    failure_rate: float
    distribution: dict[int, int]
    total_games: int


def evaluate_strategy(
    scorer: Scorer,
    word_list: WordList,
    answers: list[str] | None = None,
    starting_word: str | None = None,
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    show_progress: bool = True,
) -> EvaluationResult:
    """
    Evaluate a scoring strategy against a set of answers.
    
    Args:
        scorer: Scoring strategy to evaluate
        word_list: Word list to use during gameplay
        answers: List of answer words to test against (defaults to all words in word_list)
        starting_word: Starting word to use (if None, uses best guess from scorer)
        max_guesses: Maximum allowed guesses
        fail_penalty: Score penalty for failures
        show_progress: Whether to show progress bar
        
    Returns:
        EvaluationResult with statistics
    """
    if answers is None:
        answers = list(word_list.keys())
    
    player = Player(scorer=scorer)
    
    distribution: dict[int, int] = {}
    total_score = 0.0
    failures = 0
    
    iterator = tqdm(answers, desc=f"Evaluating strategy") if show_progress else answers
    for answer in iterator:
        result = simulate_game(
            answer=answer,
            player=player,
            word_list=word_list,
            starting_word=starting_word,
            max_guesses=max_guesses,
        )
        
        guesses = result.guesses
        if guesses > max_guesses:
            failures += 1
            total_score += fail_penalty
            distribution[max_guesses + 1] = distribution.get(max_guesses + 1, 0) + 1
        else:
            total_score += guesses
            distribution[guesses] = distribution.get(guesses, 0) + 1
    
    total_games = len(answers)
    return EvaluationResult(
        starting_word=starting_word or "auto",
        average_score=total_score / total_games if total_games > 0 else 0,
        failure_rate=failures / total_games if total_games > 0 else 0,
        distribution=dict(sorted(distribution.items())),
        total_games=total_games,
    )

