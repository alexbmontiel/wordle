from wordle_helper.core.constraints import Constraints, parse_response


def ismatch(word: str, word_chars: frozenset[str], constraints: Constraints) -> bool:
    """Check if word satisfies all constraints using set operations."""
    # Check absent first (most common rejection) - O(1) set intersection
    if constraints.absent & word_chars:
        return False

    # Check present letters exist - O(1) subset check
    if not constraints.present <= word_chars:
        return False

    # Check greens: exact position matches
    for pos, char in constraints.exact:
        if word[pos] != char:
            return False

    # Check absent_at: yellows can't be at their guessed position
    for pos, char in constraints.absent_at:
        if word[pos] == char:
            return False

    return True


def filter_word_list(
    guess: str,
    result: str,
    word_list: dict[str, tuple[float, frozenset[str]]],
) -> dict[str, tuple[float, frozenset[str]]]:
    """Filter word list based on guess and response."""
    constraints = parse_response(guess, result)
    return {
        word: data
        for word, data in word_list.items()
        if ismatch(word, data[1], constraints)
    }
