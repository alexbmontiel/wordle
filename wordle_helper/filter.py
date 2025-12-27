from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(slots=True)
class Constraints:
    exact: list[tuple[int, str]] = field(default_factory=list)  # greens: (pos, char)
    present: set[str] = field(default_factory=set)  # must exist somewhere (yellows)
    absent_at: list[tuple[int, str]] = field(default_factory=list)  # yellow positions
    absent: set[str] = field(default_factory=set)  # can't exist anywhere (greys)


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
            # else: duplicate letter marked grey - ignore (already confirmed elsewhere)
        else:
            raise ValueError(f"Unexpected code: {code}")

    return Constraints(exact=exact, present=present, absent_at=absent_at, absent=absent)


def ismatch(word: str, constraints: Constraints) -> bool:
    """Check if word satisfies all constraints."""
    # Check greens: exact position matches
    for pos, char in constraints.exact:
        if word[pos] != char:
            return False

    # Check absent: letters that can't exist
    for char in constraints.absent:
        if char in word:
            return False

    # Check present: letters that must exist somewhere
    for char in constraints.present:
        if char not in word:
            return False

    # Check absent_at: yellows can't be at their guessed position
    for pos, char in constraints.absent_at:
        if word[pos] == char:
            return False

    return True


def filter_word_list(
    guess: str,
    result: str,
    word_list: OrderedDict[str, float],
) -> OrderedDict[str, float]:
    """Filter word list based on guess and response."""
    constraints = parse_response(guess, result)
    return OrderedDict(
        (word, freq) for word, freq in word_list.items() if ismatch(word, constraints)
    )
