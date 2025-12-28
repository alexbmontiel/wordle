"""Constraint representation for Wordle filtering."""

from dataclasses import dataclass


@dataclass(slots=True)
class Constraints:
    """
    Represents constraints derived from a guess result.
    
    Attributes:
        exact: Tuple of (position, character) for green letters (correct position)
        present: Set of characters that must exist somewhere in the word (yellow)
        absent_at: Tuple of (position, character) for yellow letters (wrong position)
        absent: Set of characters that cannot exist anywhere (grey)
    """
    exact: tuple[tuple[int, str], ...]  # greens: (pos, char) as tuple for speed
    present: frozenset[str]  # must exist somewhere (yellows)
    absent_at: tuple[tuple[int, str], ...]  # yellow positions as tuple for speed
    absent: frozenset[str]  # can't exist anywhere (greys)


def parse_response(guess: str, result: str) -> Constraints:
    """
    Convert guess + response into pre-computed constraints.
    
    Args:
        guess: The guessed word (5 characters)
        result: The result string (5 characters: G/Y/N)
        
    Returns:
        Constraints object representing the filtering rules
    """
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

