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

### Frequency Transformation Commands

The helper includes tools for analyzing and optimizing frequency transformations:

```bash
# Visualize frequency transformation with sigmoid curve
wordle freq plot --k 10.0 --x0 0.5

# Show words near the sigmoid cutoff point
wordle freq cutoff --k 10.0 --x0 0.5 --words 20

# Optimize sigmoid parameters against an answer list
wordle freq optimize wordle_answers.txt --quick
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on frequency transformations.

## Strategy

The bot uses **frequency-weighted information gain (entropy)** to select optimal guesses:

1. **Partitioning**: For each candidate guess, partition remaining words by what result they'd produce (243 possible outcomes: G/Y/N for each position)
2. **Information gain**: Calculate expected bits of information gained (Shannon entropy), weighted by word frequency (common words matter more)
3. **Selection**: Choose the guess that maximizes expected information gain

This approach prioritizes eliminating common words (likely Wordle answers) over obscure ones.

For more details on how the strategy works, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```

## Architecture

For detailed documentation on how the codebase works, see [ARCHITECTURE.md](ARCHITECTURE.md). This includes:
- Module structure and data flow
- Explanation of the result matrix optimization
- How scoring strategies work
- Usage examples

## Python API

```python
from wordle_helper.data import create_word_list, WordListConfig
from wordle_helper.cli import simulated_game
from wordle_helper.evaluation.benchmark import evaluate_strategy
from wordle_helper.scoring.strategies import InformationGainScorer

# Create word list
word_list = create_word_list(WordListConfig(max_words=10000))

# Simulate a game (simulated_game creates player internally)
result = simulated_game("CRANE", starting_word="SLATE", word_list=word_list, verbose=False)
print(f"Solved in {result['guesses']} guesses")

# Evaluate a starting word across all answers
scorer = InformationGainScorer()
stats = evaluate_strategy(scorer, word_list, starting_word="SLATE")
print(f"Average: {stats.average_score:.2f}, Failures: {stats.failure_rate:.1%}")
```

For more details, see [ARCHITECTURE.md](ARCHITECTURE.md).
