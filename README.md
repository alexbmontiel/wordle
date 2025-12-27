# Wordle Helper

A CLI tool that helps you solve Wordle puzzles by suggesting optimal guesses based on remaining possible words.

## Installation

```bash
git clone https://github.com/yourusername/wordle.git
cd wordle
pip install .
```

## Usage

Run the helper:

```bash
wordle
```

The tool will:
1. Suggest the best word to guess (based on word frequency)
2. After you enter your guess and the result, it filters the word list
3. Repeat until solved or 6 guesses are exhausted

### Result Format

After entering your guess in Wordle, enter the result using:
- `G` = Green (correct letter, correct position)
- `Y` = Yellow (correct letter, wrong position)
- `N` = Grey (letter not in word)

### Example Session

```
=== Wordle Helper CLI ===

Recommended guess: ABOUT (8547 words remaining)
Guess 1/6 (or 'q' to quit): CRANE
Enter result (G=green, Y=yellow, N=nothing): NGGYN

Recommended guess: TRAIN (12 words remaining)
Guess 2/6 (or 'q' to quit): TRAIN
Enter result (G=green, Y=yellow, N=nothing): GGGGG

Solved in 2 guesses!
```

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```
