"""Partitioning logic for scoring guesses."""

from wordle_helper.core.result import compute_result
from wordle_helper.data.word_list import WordList


def partition_by_result(
    guess: str,
    word_list: WordList,
) -> dict[str, tuple[list[str], float]]:
    """
    Partition words by what result they'd produce for this guess.

    Returns a dict mapping result string -> (list of words, sum of frequencies).
    At most 243 buckets (3^5 possible results: G/Y/N for each of 5 positions).

    Args:
        guess: The guess word to evaluate
        word_list: Current word list

    Returns:
        Dict mapping result string (e.g., "GGGGG", "NYGYN") -> (list of words, frequency sum)
    """
    buckets: dict[str, tuple[list[str], float]] = {}
    for word, (freq, _) in word_list.items():
        result = compute_result(guess, word)
        if result in buckets:
            words, freq_sum = buckets[result]
            words.append(word)
            buckets[result] = (words, freq_sum + freq)
        else:
            buckets[result] = ([word], freq)
    return buckets

