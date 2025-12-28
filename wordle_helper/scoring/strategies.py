"""Concrete scoring strategy implementations."""

import math
from wordle_helper.scoring.base import Scorer
from wordle_helper.scoring.partition import partition_by_result
from wordle_helper.data.word_list import WordList


class InformationGainScorer(Scorer):
    """
    Score guesses using information gain (entropy).
    
    Calculates expected information gain (entropy) for a guess, weighted by word frequency.
    Higher scores indicate guesses that provide more information on average.
    """
    
    def score(self, guess: str, word_list: WordList) -> float:
        """
        Calculate expected information gain (entropy) for a guess, weighted by word frequency.

        Higher is better - measures how much uncertainty is reduced on average,
        giving more weight to eliminating common words.
        """
        buckets = partition_by_result(guess, word_list)
        total_freq = sum(data[0] for data in word_list.values())

        if total_freq <= 0:
            return 0.0

        # Entropy weighted by frequency: sum(p * log2(1/p))
        # where p = bucket_freq_sum / total_freq
        entropy = 0.0
        for _, freq_sum in buckets.values():
            p = freq_sum / total_freq
            if p > 0:
                entropy += p * math.log2(1 / p)

        return entropy


class ExpectedRemainingScorer(Scorer):
    """
    Score guesses using expected remaining words.
    
    Calculates expected number of remaining words after a guess (frequency-weighted).
    Lower scores indicate better guesses (fewer words remaining).
    """
    
    def score(self, guess: str, word_list: WordList) -> float:
        """
        Score a guess by expected remaining words after guessing (frequency-weighted).

        Note: Lower is better for this scorer, so this returns negative expected remaining.
        This makes it compatible with the Scorer interface (higher = better).
        """
        buckets = partition_by_result(guess, word_list)
        total_freq = sum(data[0] for data in word_list.values())

        if total_freq == 0:
            return 0.0

        # Expected value weighted by frequency
        # sum of (p_bucket * freq_bucket) where p_bucket = freq_bucket / total_freq
        expected_remaining = sum(freq_sum ** 2 for _, freq_sum in buckets.values()) / total_freq
        
        # Return negative so higher scores are better (compatible with Scorer interface)
        return -expected_remaining

