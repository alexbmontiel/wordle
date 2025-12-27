from collections import OrderedDict

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .corpus import create_word_list
from .filter import filter_word_list

console = Console()
MAX_GUESSES = 6


COLOR_KEY = {
    "G": ("green", "correct position"),
    "Y": ("yellow", "wrong position"),
    "N": ("grey39", "not in word")
}


def recommended_guess(word_list: OrderedDict[str, float]):
    if word_list:
        return next(iter(word_list.keys()), None)
    return None


def format_result(guess, result):
    text = Text()
    for g, c in zip(guess, result):
        color = COLOR_KEY.get(c, ("white",))[0]
        text.append(g.upper(), style=color)
    return text


def display_history():
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


def live_game():
    global history
    history = []
    word_list = create_word_list()
    guess_num = 0

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
        display_history()

        if result == "GGGGG":
            console.print(f"\n[bold green]Solved in {guess_num} guesses![/bold green]\n")
            return

        word_list = filter_word_list(guess, result, word_list)
        display_key()

    console.print("\n[bold red]Game over! Out of guesses.[/bold red]\n")