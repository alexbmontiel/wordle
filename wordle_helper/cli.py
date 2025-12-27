from rich.console import Console
from rich.table import Table
from rich.text import Text

from .corpus import create_word_list
from .filter import filter_word_list
from .strategy import compute_result, best_guess, information_gain

console = Console()
MAX_GUESSES = 6


COLOR_KEY = {
    "G": ("green", "correct position"),
    "Y": ("yellow", "wrong position"),
    "N": ("grey39", "not in word")
}


def recommended_guess(word_list: dict[str, tuple[float, frozenset[str]]]):
    if word_list:
        return next(iter(word_list.keys()), None)
    return None


def format_result(guess, result):
    text = Text()
    for g, c in zip(guess, result):
        color = COLOR_KEY.get(c, ("white",))[0]
        text.append(g.upper(), style=color)
    return text


def get_top_candidates(word_list: dict[str, tuple[float, frozenset[str]]], n: int = 5) -> list[tuple[str, float]]:
    """Get top n candidates by information gain."""
    if len(word_list) <= n:
        candidates = list(word_list.keys())
    else:
        candidates = list(word_list.keys())[:50]  # Limit for speed

    scored = [(w, information_gain(w, word_list)) for w in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def simulated_game(
    answer: str,
    starting_word: str | None = None,
    word_list: dict[str, tuple[float, frozenset[str]]] | None = None,
    max_guesses: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Simulate a game with optional visual output.

    Args:
        answer: The target word
        starting_word: First guess (if None, bot picks optimal)
        word_list: Word list to use (if None, creates default)
        max_guesses: Maximum allowed guesses
        verbose: If True, print visual output

    Returns:
        Dict with game results:
        - guesses: Number of guesses taken
        - success: Whether the game was won
        - history: List of (guess, result) tuples
        - choices: List of top candidates considered each turn
    """
    if word_list is None:
        word_list = create_word_list()

    remaining = word_list.copy()
    answer = answer.upper()
    history = []
    choices = []

    if verbose:
        console.print("\n[bold underline]=== Simulating Game ===[/bold underline]")
        console.print(f"Answer: [bold cyan]{answer}[/bold cyan]\n")

    # Determine first guess
    if starting_word:
        guess = starting_word.upper()
    else:
        guess = best_guess(remaining)
        if guess is None:
            return {
                "guesses": max_guesses + 1,
                "success": False,
                "history": [],
                "choices": [],
            }

    for turn in range(1, max_guesses + 1):
        result = compute_result(guess, answer)
        history.append((guess, result))

        # Get top candidates before filtering (for logging)
        top = get_top_candidates(remaining, n=5)
        choices.append({"turn": turn, "guess": guess, "top_candidates": top})

        if verbose:
            # Show turn info
            console.print(f"[bold]Turn {turn}/{max_guesses}[/bold]")
            console.print("  Guess: ", end="")
            console.print(format_result(guess, result))
            console.print(f"  Remaining words: {len(remaining)}")

            # Show top candidates
            if len(remaining) > 1:
                console.print("  Top candidates:")
                for word, score in top[:5]:
                    marker = " [bold green]<< chosen[/bold green]" if word == guess else ""
                    console.print(f"    {word}: {score:.3f} bits{marker}")
            console.print()

        if result == "GGGGG":
            if verbose:
                console.print(f"[bold green]Solved in {turn} guess{'es' if turn > 1 else ''}![/bold green]\n")
            return {
                "guesses": turn,
                "success": True,
                "history": history,
                "choices": choices,
            }

        # Filter and pick next guess
        remaining = filter_word_list(guess, result, remaining)

        if not remaining:
            if verbose:
                console.print("[bold red]No valid words remaining![/bold red]\n")
            return {
                "guesses": max_guesses + 1,
                "success": False,
                "history": history,
                "choices": choices,
            }

        next_guess = best_guess(remaining)
        if next_guess is None:
            break
        guess = next_guess

    # Failed
    if verbose:
        console.print(f"[bold red]Failed! Answer was {answer}[/bold red]\n")

    return {
        "guesses": max_guesses + 1,
        "success": False,
        "history": history,
        "choices": choices,
    }


def display_history(history: list[tuple[str, str]]):
    if not history:
        return
    table = Table(title="History")
    table.add_column("Guess", justify="center")
    for guess, result in history:
        table.add_row(format_result(guess, result))
    console.print("\n", table)


def display_key():
    console.print("\nColor key:")
    key_text = Text()
    for k, (color, desc) in COLOR_KEY.items():
        key_text.append(f"{k}: {desc}  ", style=color)
    console.print(key_text)
    console.print()


def live_game(
    word_list: dict[str, tuple[float, frozenset[str]]] | None = None
):
    history = []
    guess_num = 0

    if word_list is None:
        word_list = create_word_list()

    console.print("\n[bold underline]=== Wordle Helper CLI ===[/bold underline]\n")
    while guess_num < MAX_GUESSES:
        rec = recommended_guess(word_list)
        if rec:
            console.print(f"Recommended guess: [bold]{rec}[/bold] ({len(word_list)} words remaining)")
        else:
            console.print("[red]No valid words remaining![/red]")
            break

        guess = console.input(f"Guess {guess_num + 1}/{MAX_GUESSES} (or 'q' to quit): ").upper()
        if guess.lower() == 'q':
            return

        if len(guess) != 5 or not guess.isalpha():
            console.print("Please enter a valid 5-letter word.", style="red")
            continue

        result = console.input("Enter result (G=green, Y=yellow, N=nothing): ").upper()
        if len(result) != 5 or any(c not in COLOR_KEY for c in result):
            console.print("Invalid result format. Example: GYNNY", style="red")
            continue

        guess_num += 1
        history.append((guess, result))
        display_history(history)

        if result == "GGGGG":
            console.print(f"\n[bold green]Solved in {guess_num} guesses![/bold green]\n")
            return

        word_list = filter_word_list(guess, result, word_list)
        display_key()

    console.print("\n[bold red]Game over! Out of guesses.[/bold red]\n")