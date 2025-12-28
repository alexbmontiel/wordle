"""CLI interface for Wordle helper."""

from rich.console import Console
from rich.table import Table
from rich.text import Text

from wordle_helper.data import create_word_list, WordList, WordListConfig
from wordle_helper.filtering.filter import filter_word_list
from wordle_helper.core.result import compute_result
from wordle_helper.game.player import Player
from wordle_helper.game.state import GameState
from wordle_helper.game.engine import play_turn, is_solved
from wordle_helper.scoring.strategies import InformationGainScorer
from wordle_helper.evaluation.simulator import simulate_game as sim_game, GameResult

console = Console()
MAX_GUESSES = 6


COLOR_KEY = {
    "G": ("green", "correct position"),
    "Y": ("yellow", "wrong position"),
    "N": ("grey39", "not in word")
}


def format_result(guess, result):
    """Format a guess result with colors."""
    text = Text()
    for g, c in zip(guess, result):
        color = COLOR_KEY.get(c, ("white",))[0]
        text.append(g.upper(), style=color)
    return text


def simulated_game(
    answer: str,
    starting_word: str | None = None,
    word_list: WordList | None = None,
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
        word_list = create_word_list(WordListConfig(max_words=10000))
    
    # Create player with information gain scorer
    scorer = InformationGainScorer()
    player = Player(scorer=scorer)
    
    # Use the new simulator
    result = sim_game(
        answer=answer,
        player=player,
        word_list=word_list,
        starting_word=starting_word,
        max_guesses=max_guesses,
    )
    
    if verbose:
        console.print("\n[bold underline]=== Simulating Game ===[/bold underline]")
        console.print(f"Answer: [bold cyan]{answer}[/bold cyan]\n")
        
        for choice in result.choices:
            turn = choice["turn"]
            guess = choice["guess"]
            top_candidates = choice["top_candidates"]
            remaining_count = choice["remaining_count"]
            
            # Find result for this guess
            result_str = None
            for g, r in result.history:
                if g == guess:
                    result_str = r
                    break
            
            if result_str:
                console.print(f"[bold]Turn {turn}/{max_guesses}[/bold]")
                console.print("  Guess: ", end="")
                console.print(format_result(guess, result_str))
                console.print(f"  Remaining words: {remaining_count}")
                
                if len(top_candidates) > 1:
                    console.print("  Top candidates:")
                    for word, score in top_candidates[:5]:
                        marker = " [bold green]<< chosen[/bold green]" if word == guess else ""
                        console.print(f"    {word}: {score:.3f} bits{marker}")
                console.print()
        
        if result.success:
            console.print(f"[bold green]Solved in {result.guesses} guess{'es' if result.guesses > 1 else ''}![/bold green]\n")
        else:
            console.print(f"[bold red]Failed! Answer was {answer}[/bold red]\n")
    
    # Convert GameResult to dict for backward compatibility
    return {
        "guesses": result.guesses,
        "success": result.success,
        "history": result.history,
        "choices": result.choices,
    }


def display_history(history: list[tuple[str, str]]):
    """Display guess history in a table."""
    if not history:
        return
    table = Table(title="History")
    table.add_column("Guess", justify="center")
    for guess, result in history:
        table.add_row(format_result(guess, result))
    console.print("\n", table)


def display_key():
    """Display the color key."""
    console.print("\nColor key:")
    key_text = Text()
    for k, (color, desc) in COLOR_KEY.items():
        key_text.append(f"{k}: {desc}  ", style=color)
    console.print(key_text)
    console.print()


def live_game(word_list: WordList | None = None, player: Player | None = None):
    """
    Play an interactive live game.
    
    Args:
        word_list: Word list to use (if None, creates default)
        player: Player with scoring strategy (if None, uses InformationGainScorer)
    """
    if word_list is None:
        word_list = create_word_list(WordListConfig(max_words=10000))
    
    if player is None:
        scorer = InformationGainScorer()
        player = Player(scorer=scorer)
    
    state = GameState(remaining_words=word_list)
    console.print("\n[bold underline]=== Wordle Helper CLI ===[/bold underline]\n")
    
    while state.turn < MAX_GUESSES:
        # Analyze state and get recommendation
        strategy_state = player.analyze_state(state, top_n=1)
        if strategy_state.top_candidates:
            rec = strategy_state.top_candidates[0][0]
            console.print(f"Recommended guess: [bold]{rec}[/bold] ({strategy_state.remaining_count} words remaining)")
        else:
            console.print("[red]No valid words remaining![/red]")
            break
        
        guess = console.input(f"Guess {state.turn + 1}/{MAX_GUESSES} (or 'q' to quit): ").upper()
        if guess.lower() == 'q':
            return
        
        if len(guess) != 5 or not guess.isalpha():
            console.print("Please enter a valid 5-letter word.", style="red")
            continue
        
        result = console.input("Enter result (G=green, Y=yellow, N=nothing): ").upper()
        if len(result) != 5 or any(c not in COLOR_KEY for c in result):
            console.print("Invalid result format. Example: GYNNY", style="red")
            continue
        
        # Update state
        filtered_words = filter_word_list(guess, result, state.remaining_words)
        state = play_turn(state, guess, result, filtered_words)
        display_history(state.history)
        
        if is_solved(result):
            console.print(f"\n[bold green]Solved in {state.turn} guesses![/bold green]\n")
            return
        
        display_key()
    
    console.print("\n[bold red]Game over! Out of guesses.[/bold red]\n")
