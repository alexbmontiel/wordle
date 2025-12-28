"""Game simulation for testing strategies."""

from dataclasses import dataclass
from wordle_helper.core.result import compute_result
from wordle_helper.filtering.filter import filter_word_list
from wordle_helper.game.state import GameState
from wordle_helper.game.engine import play_turn, is_solved
from wordle_helper.game.player import Player
from wordle_helper.data.word_list import WordList


@dataclass
class GameResult:
    """
    Results from a simulated game.
    
    Attributes:
        guesses: Number of guesses taken (or max_guesses + 1 if failed)
        success: Whether the game was won
        history: List of (guess, result) tuples
        choices: List of strategy states at each turn (for analysis)
    """
    guesses: int
    success: bool
    history: list[tuple[str, str]]
    choices: list[dict]  # Strategy state at each turn


def simulate_game(
    answer: str,
    player: Player,
    word_list: WordList,
    starting_word: str | None = None,
    max_guesses: int = 6,
) -> GameResult:
    """
    Simulate a game with a player strategy.
    
    Args:
        answer: The target word
        player: Player with scoring strategy
        word_list: Initial word list
        starting_word: First guess (if None, player chooses)
        max_guesses: Maximum allowed guesses
        
    Returns:
        GameResult with game outcome and details
    """
    answer = answer.upper()
    state = GameState(remaining_words=word_list)
    choices = []
    
    # Determine first guess
    if starting_word:
        guess = starting_word.upper()
    else:
        guess = player.choose_guess(state)
        if guess is None:
            return GameResult(
                guesses=max_guesses + 1,
                success=False,
                history=[],
                choices=[],
            )
    
    history = []
    for turn in range(1, max_guesses + 1):
        result = compute_result(guess, answer)
        history.append((guess, result))
        
        # Analyze strategy state before filtering (for research)
        strategy_state = player.analyze_state(state)
        choices.append({
            "turn": turn,
            "guess": guess,
            "top_candidates": strategy_state.top_candidates[:5],
            "remaining_count": strategy_state.remaining_count,
        })
        
        if is_solved(result):
            return GameResult(
                guesses=turn,
                success=True,
                history=history,
                choices=choices,
            )
        
        # Filter word list and update state
        filtered_words = filter_word_list(guess, result, state.remaining_words)
        state = play_turn(state, guess, result, filtered_words)
        
        if not state.remaining_words:
            return GameResult(
                guesses=max_guesses + 1,
                success=False,
                history=history,
                choices=choices,
            )
        
        guess = player.choose_guess(state)
        if guess is None:
            break
    
    # Failed
    return GameResult(
        guesses=max_guesses + 1,
        success=False,
        history=state.history,
        choices=choices,
    )

