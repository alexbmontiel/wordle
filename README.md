# Wordle Helper

A CLI tool that helps you solve Wordle puzzles using information-theoretic optimal guesses.

## Installation

```bash
git clone https://github.com/yourusername/wordle.git
cd wordle
pip install .
```

## Usage

### Live Game (Interactive)

Play interactively with the helper suggesting optimal guesses:

```bash
wordle              # Default: play live
wordle live         # Explicit command
wordle --n 5000     # Limit to top 5000 most common words
```

The tool will:
1. Suggest the best word to guess (maximizing information gain)
2. After you enter your guess and the Wordle result, it filters possibilities
3. Repeat until solved or 6 guesses are exhausted

### Simulated Game

Watch the bot play against a known answer:

```bash
wordle simulate CRANE                 # Bot picks optimal starting word
wordle simulate CRANE -s SLATE        # Specify starting word
wordle simulate CRANE -q              # Quiet mode (just show result)
wordle simulate CRANE -s SLATE -q     # Both options
```

**Options:**
- `-s, --start` - Starting word (bot picks if not provided)
- `-n` - Limit word list to top N words (default: 10000)
- `-q, --quiet` - Suppress visual output for batch testing

**Verbose output shows:**
- Colored guess results (green/yellow/grey)
- Remaining word count after each guess
- Top 5 candidates with information gain scores

### Result Format

When playing live, enter results using:
- `G` = Green (correct letter, correct position)
- `Y` = Yellow (correct letter, wrong position)
- `N` = Grey (letter not in word)

### Example: Live Session

```
=== Wordle Helper CLI ===

Recommended guess: SLATE (10000 words remaining)
Guess 1/6 (or 'q' to quit): SLATE
Enter result (G=green, Y=yellow, N=nothing): NYGYN

Recommended guess: TRAIL (42 words remaining)
Guess 2/6 (or 'q' to quit): TRAIL
Enter result (G=green, Y=yellow, N=nothing): GGGGG

Solved in 2 guesses!
```

### Example: Simulated Game

```bash
$ wordle simulate CRANE -s SLATE
```

```
=== Simulating Game ===
Answer: CRANE

Turn 1/6
  Guess: SLATE
  Remaining words: 10000
  Top candidates:
    LATER: 5.578 bits
    THEIR: 5.514 bits
    ...

Turn 2/6
  Guess: GRACE
  Remaining words: 61
  Top candidates:
    GRACE: 2.884 bits << chosen
    ...

Turn 3/6
  Guess: CRANE
  Remaining words: 4

Solved in 3 guesses!
```

## Strategy

The bot uses **frequency-weighted entropy** to select optimal guesses:

1. **Partitioning**: For each candidate guess, partition remaining words by what result they'd produce (243 possible outcomes: G/Y/N for each position)
2. **Information gain**: Calculate expected bits of information gained, weighted by word frequency (common words matter more)
3. **Selection**: Choose the guess that maximizes expected information gain

This approach prioritizes eliminating common words (likely Wordle answers) over obscure ones.

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```

## Python API

```python
from wordle_helper import corpus
from wordle_helper.cli import simulated_game
from wordle_helper.strategy import information_gain, evaluate_starting_word

# Create word list
word_list = corpus.create_word_list()

# Simulate a game
result = simulated_game("CRANE", starting_word="SLATE", verbose=False)
print(f"Solved in {result['guesses']} guesses")

# Evaluate a starting word across all answers
stats = evaluate_starting_word("SLATE", word_list)
print(f"Average: {stats['average_score']:.2f}, Failures: {stats['failure_rate']:.1%}")
```
