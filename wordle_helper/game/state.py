"""Game state and strategy state representations."""

from dataclasses import dataclass, field
from wordle_helper.data.word_list import WordList


@dataclass
class GameState:
    """
    Represents the current state of a Wordle game.
    
    Attributes:
        remaining_words: Words that could still be the answer
        history: List of (guess, result) tuples
        turn: Current turn number (1-indexed)
    """
    remaining_words: WordList
    history: list[tuple[str, str]] = field(default_factory=list)
    turn: int = 0
    
    def with_turn(self, guess: str, result: str) -> "GameState":
        """
        Create a new GameState after playing a turn.
        
        Args:
            guess: The guess made
            result: The result string (G/Y/N)
            
        Returns:
            New GameState with updated history and turn
        """
        return GameState(
            remaining_words=self.remaining_words,
            history=self.history + [(guess, result)],
            turn=self.turn + 1,
        )


@dataclass
class StrategyState:
    """
    Represents the strategy's analysis of the current game state.
    
    Attributes:
        top_candidates: List of (word, score) tuples for top candidate guesses
        remaining_count: Number of remaining valid words
        last_guess: Most recent guess (if any)
        last_result: Most recent result (if any)
    """
    top_candidates: list[tuple[str, float]]
    remaining_count: int
    last_guess: str | None = None
    last_result: str | None = None
    
    def get_top_candidates(self, n: int) -> list[tuple[str, float]]:
        """Get top N candidates."""
        return self.top_candidates[:n]
    
    def summary(self) -> dict:
        """Get summary dictionary of strategy state."""
        return {
            "remaining_count": self.remaining_count,
            "top_candidate": self.top_candidates[0][0] if self.top_candidates else None,
            "top_score": self.top_candidates[0][1] if self.top_candidates else None,
            "last_guess": self.last_guess,
            "last_result": self.last_result,
        }

