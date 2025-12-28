"""Core game engine functions."""

from wordle_helper.data.word_list import WordList
from wordle_helper.game.state import GameState


def is_solved(result: str) -> bool:
    """
    Check if a result string indicates the game is solved.
    
    Args:
        result: Result string (5 characters of G/Y/N)
        
    Returns:
        True if result is all G's (solved)
    """
    return result == "GGGGG"


def play_turn(state: GameState, guess: str, result: str, filtered_words: WordList) -> GameState:
    """
    Play a turn and return updated game state.
    
    This is a pure function that creates a new GameState from the old one.
    The filtering logic should be handled separately (see filtering module)
    and passed in as filtered_words.
    
    Args:
        state: Current game state
        guess: The guess made
        result: The result string (G/Y/N)
        filtered_words: Word list after applying constraints (already filtered)
        
    Returns:
        New GameState with updated remaining_words, history, and turn
    """
    return GameState(
        remaining_words=filtered_words,
        history=state.history + [(guess, result)],
        turn=state.turn + 1,
    )

