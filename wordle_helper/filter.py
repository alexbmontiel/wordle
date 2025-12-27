from dataclasses import dataclass, field


@dataclass(slots=True)
class Constraints:
    exact: tuple[tuple[int, str], ...]  # greens: (pos, char) as tuple for speed
    present: frozenset[str]  # must exist somewhere (yellows)
    absent_at: tuple[tuple[int, str], ...]  # yellow positions as tuple for speed
    absent: frozenset[str]  # can't exist anywhere (greys)


def parse_response(guess: str, result: str) -> Constraints:
    """Convert guess + response into pre-computed constraints."""
    exact = []
    present = set()
    absent_at = []
    absent = set()

    confirmed_letters = set()
    for i, (char, code) in enumerate(zip(guess, result)):
        if code == "G":
            exact.append((i, char))
            confirmed_letters.add(char)
        elif code == "Y":
            present.add(char)
            absent_at.append((i, char))
            confirmed_letters.add(char)
        elif code == "N":
            if char not in confirmed_letters:
                absent.add(char)
        else:
            raise ValueError(f"Unexpected code: {code}")

    return Constraints(
        exact=tuple(exact),
        present=frozenset(present),
        absent_at=tuple(absent_at),
        absent=frozenset(absent),
    )


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
