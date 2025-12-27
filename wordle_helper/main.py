import typer
from typing import Optional

from .cli import live_game, simulated_game
from .corpus import create_word_list, filter_words

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, n: int = typer.Option(10000, help="Limit word list to top N words")):
    """Wordle helper CLI. Run without arguments to play live."""
    if ctx.invoked_subcommand is None:
        word_list = create_word_list()
        word_list = filter_words(word_list, n)
        live_game(word_list)


@app.command()
def live(n: int = typer.Option(10000, help="Limit word list to top N words")):
    """Play an interactive game with the Wordle helper."""
    word_list = create_word_list()
    word_list = filter_words(word_list, n)
    live_game(word_list)


@app.command()
def simulate(
    answer: str = typer.Argument(..., help="The target word to solve"),
    starting_word: Optional[str] = typer.Option(None, "--start", "-s", help="Starting word (bot picks if not provided)"),
    n: int = typer.Option(10000, help="Limit word list to top N words"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress visual output"),
):
    """Simulate a game against a known answer."""
    word_list = create_word_list()
    word_list = filter_words(word_list, n)
    result = simulated_game(
        answer=answer,
        starting_word=starting_word,
        word_list=word_list,
        verbose=not quiet,
    )
    if quiet:
        status = "Won" if result["success"] else "Lost"
        typer.echo(f"{answer}: {result['guesses']} guesses ({status})")


if __name__ == "__main__":
    app()