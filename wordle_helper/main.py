"""Main CLI entry point for Wordle helper."""

import typer
from typing import Optional
from pathlib import Path

from wordle_helper.cli import live_game, simulated_game
from wordle_helper.data import create_word_list, WordListConfig, FrequencyConfig
from wordle_helper.scoring.strategies import InformationGainScorer
from wordle_helper.game.player import Player

app = typer.Typer()
freq_app = typer.Typer(help="Frequency transformation commands")
app.add_typer(freq_app, name="freq")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, n: int = typer.Option(10000, help="Limit word list to top N words")):
    """Wordle helper CLI. Run without arguments to play live."""
    if ctx.invoked_subcommand is None:
        word_list = create_word_list(WordListConfig(max_words=n))
        live_game(word_list)


@app.command()
def live(n: int = typer.Option(10000, help="Limit word list to top N words")):
    """Play an interactive game with the Wordle helper."""
    word_list = create_word_list(WordListConfig(max_words=n))
    live_game(word_list)


@app.command()
def simulate(
    answer: str = typer.Argument(..., help="The target word to solve"),
    starting_word: Optional[str] = typer.Option(None, "--start", "-s", help="Starting word (bot picks if not provided)"),
    n: int = typer.Option(10000, help="Limit word list to top N words"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress visual output"),
):
    """Simulate a game against a known answer."""
    word_list = create_word_list(WordListConfig(max_words=n))
    result = simulated_game(
        answer=answer,
        starting_word=starting_word,
        word_list=word_list,
        verbose=not quiet,
    )
    if quiet:
        status = "Won" if result["success"] else "Lost"
        typer.echo(f"{answer}: {result['guesses']} guesses ({status})")


@freq_app.command("plot")
def freq_plot(
    k: float = typer.Option(10.0, "--k", "-k", help="Sigmoid steepness parameter"),
    x0: float = typer.Option(0.5, "--x0", "-x", help="Sigmoid midpoint (0-1)"),
    n: int = typer.Option(10000, help="Limit word list to top N words"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save plot to file"),
):
    """Visualize frequency transformation with sigmoid curve."""
    from wordle_helper.data.frequency import plot_frequency_transform
    import matplotlib.pyplot as plt

    word_list = create_word_list(WordListConfig(max_words=n))
    fig = plot_frequency_transform(word_list, k=k, x0=x0)

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        typer.echo(f"Saved plot to {output}")
    else:
        plt.show()


@freq_app.command("cutoff")
def freq_cutoff(
    k: float = typer.Option(10.0, "--k", "-k", help="Sigmoid steepness parameter"),
    x0: float = typer.Option(0.5, "--x0", "-x", help="Sigmoid midpoint (0-1)"),
    n: int = typer.Option(10000, help="Limit word list to top N words"),
    n_words: int = typer.Option(20, "--words", "-w", help="Number of words to show"),
):
    """Show words near the sigmoid cutoff point."""
    from wordle_helper.data.frequency import find_cutoff_words

    word_list = create_word_list(WordListConfig(max_words=n))
    cutoff_words = find_cutoff_words(word_list, k=k, x0=x0, n_words=n_words)

    typer.echo(f"\nWords near cutoff (k={k}, x0={x0}):\n")
    typer.echo(f"{'Word':<10} {'Raw Freq':<12} {'Weight':<8}")
    typer.echo("-" * 32)

    for word, freq, weight in sorted(cutoff_words, key=lambda x: -x[2]):
        typer.echo(f"{word:<10} {freq:<12.2e} {weight:<8.3f}")


@freq_app.command("optimize")
def freq_optimize(
    answers: Path = typer.Argument(..., help="Path to answer list file (one word per line)"),
    n: int = typer.Option(10000, help="Limit word list to top N words"),
    grid_size: int = typer.Option(5, "--grid", "-g", help="Grid search resolution"),
    quick: bool = typer.Option(True, "--quick/--full", help="Quick (2-phase) or full grid search"),
):
    """Optimize sigmoid parameters against an answer list."""
    from wordle_helper.data.optimize import optimize_sigmoid_params, quick_optimize, load_answer_list

    word_list = create_word_list(WordListConfig(max_words=n))
    answer_words = load_answer_list(str(answers))
    typer.echo(f"Loaded {len(answer_words)} answer words from {answers}")

    if quick:
        result = quick_optimize(word_list, answer_words, verbose=True)
    else:
        result = optimize_sigmoid_params(
            word_list,
            answer_words,
            method="grid",
            n_grid=grid_size,
            verbose=True,
        )

    typer.echo(f"\n{'='*40}")
    typer.echo("Optimization Results:")
    typer.echo(f"{'='*40}")
    typer.echo(f"Best k (steepness): {result['best_k']:.2f}")
    typer.echo(f"Best x0 (midpoint): {result['best_x0']:.3f}")
    typer.echo(f"Best avg guesses:   {result['best_avg_guesses']:.4f}")
    typer.echo(f"Baseline avg:       {result['baseline_avg_guesses']:.4f}")
    typer.echo(f"Improvement:        {result['improvement']:.4f} guesses")


if __name__ == "__main__":
    app()
