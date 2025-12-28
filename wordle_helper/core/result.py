"""Result computation for Wordle guesses."""

from functools import lru_cache


@lru_cache(maxsize=500000)
def compute_result(guess: str, answer: str) -> str:
    """
    Compute the G/Y/N result string for a guess against an answer.
    
    Returns a 5-character string where:
    - 'G' = Green (correct letter, correct position)
    - 'Y' = Yellow (correct letter, wrong position)
    - 'N' = Grey (letter not in word)
    
    Cached for performance.
    
    Args:
        guess: The guessed word
        answer: The target word
        
    Returns:
        5-character result string (e.g., "GGGGG", "NYGYN")
    """
    result = ["N"] * 5
    answer_chars: list[str | None] = list(answer)

    # First pass: mark greens and remove matched chars from answer
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            result[i] = "G"
            answer_chars[i] = None

    # Second pass: mark yellows
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in answer_chars:
            result[i] = "Y"
            answer_chars[answer_chars.index(g)] = None

    return "".join(result)

