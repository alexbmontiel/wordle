"""Player/solver logic."""

from wordle_helper.game.state import GameState, StrategyState
from wordle_helper.scoring.base import Scorer
from wordle_helper.data.word_list import WordList


class Player:
    """
    A Wordle player that uses a scoring strategy to choose guesses.
    
    The player can analyze the current game state and choose the best guess
    according to its scorer.
    """
    
    def __init__(self, scorer: Scorer):
        """
        Initialize a player with a scoring strategy.
        
        Args:
            scorer: The scoring strategy to use for selecting guesses
        """
        self.scorer = scorer
    
    def choose_guess(self, state: GameState) -> str | None:
        """
        Choose the best guess for the current game state.
        
        Args:
            state: Current game state
            
        Returns:
            Best guess word, or None if no words remain
        """
        if not state.remaining_words:
            return None
        
        # If only one word remains, guess it
        if len(state.remaining_words) == 1:
            return next(iter(state.remaining_words))
        
        # Find best guess using scorer
        best_word = None
        best_score = float('-inf')
        
        for word in state.remaining_words:
            score = self.scorer.score(word, state.remaining_words)
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word
    
    def analyze_state(self, state: GameState, top_n: int = 10) -> StrategyState:
        """
        Analyze the current game state and return strategy analysis.
        
        This provides insight into what the strategy is thinking:
        - Top candidate guesses and their scores
        - Remaining word count
        - Last guess/result
        
        Args:
            state: Current game state
            top_n: Number of top candidates to include
            
        Returns:
            StrategyState with analysis
        """
        top_candidates = self.scorer.rank_candidates(state.remaining_words, top_n=top_n)
        
        last_guess = None
        last_result = None
        if state.history:
            last_guess, last_result = state.history[-1]
        
        return StrategyState(
            top_candidates=top_candidates,
            remaining_count=len(state.remaining_words),
            last_guess=last_guess,
            last_result=last_result,
        )

