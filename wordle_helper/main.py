import typer

from .cli import live_game

app = typer.Typer()


@app.command()
def play_game():
    live_game()


if __name__ == "__main__":
    app()