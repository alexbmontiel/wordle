"""Abstract base classes for scoring strategies."""

from abc import ABC, abstractmethod
from wordle_helper.data.word_list import WordList


class Scorer(ABC):
    """
    Abstract base class for scoring word guesses.
    
    Higher scores indicate better guesses.
    """
    
    @abstractmethod
    def score(self, guess: str, word_list: WordList) -> float:
        """
        Score a guess against a word list.
        
        Args:
            guess: The word to score
            word_list: Current word list of possible words
            
        Returns:
            Score (higher is better)
        """
        pass
    
    def rank_candidates(
        self,
        word_list: WordList,
        candidates: list[str] | None = None,
        top_n: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank candidate words by score.
        
        Args:
            word_list: Current word list
            candidates: Words to evaluate (defaults to all words in word_list)
            top_n: Return only top N candidates (None = return all)
            
        Returns:
            List of (word, score) tuples sorted by score descending
        """
        if candidates is None:
            candidates = list(word_list.keys())
        
        scored = [(word, self.score(word, word_list)) for word in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if top_n is not None:
            return scored[:top_n]
        return scored

