# Wordle Helper Architecture Documentation

This document explains how the Wordle helper codebase is structured and how all the pieces work together.

## Table of Contents

1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [Data Flow](#data-flow)
4. [Core Concepts](#core-concepts)
5. [Understanding the Result Matrix](#understanding-the-result-matrix)
6. [Module Details](#module-details)
7. [Usage Examples](#usage-examples)

## Overview

The Wordle helper uses an information-theoretic approach to solve Wordle puzzles. At its core, it:

1. **Scores guesses** by calculating how much information they provide (information gain/entropy)
2. **Filters words** based on guess results (green/yellow/grey feedback)
3. **Selects the best next guess** from remaining words
4. **Repeats** until the answer is found

The codebase is organized into clean, modular components that can be easily understood, tested, and modified.

## Module Structure

```
wordle_helper/
├── core/              # Low-level primitives (no game logic dependencies)
│   ├── result.py      # Compute G/Y/N results from guess+answer
│   └── constraints.py # Parse results into filterable constraints
│
├── data/              # Word lists and frequency handling
│   ├── corpus.py      # Load raw word lists from wordfreq
│   ├── word_list.py   # WordList type, configuration
│   ├── frequency.py   # Frequency transformation (sigmoid)
│   └── optimize.py    # Optimize frequency transform parameters
│
├── filtering/         # Apply constraints to word lists
│   └── filter.py      # Filter words based on G/Y/N results
│
├── scoring/           # Scoring strategies (pluggable)
│   ├── base.py        # Scorer abstract interface
│   ├── strategies.py  # InformationGainScorer, ExpectedRemainingScorer
│   ├── partition.py   # Partition words by result buckets
│   └── optimized.py   # Placeholder for numba-optimized scorers
│
├── game/              # Game orchestration
│   ├── state.py       # GameState, StrategyState
│   ├── engine.py      # Core game functions (play_turn, is_solved)
│   └── player.py      # Player class with scoring strategy
│
├── evaluation/        # Testing and benchmarking
│   ├── simulator.py   # Simulate games
│   ├── benchmark.py   # Evaluate strategies
│   └── optimized_evaluator.py  # FastEvaluator (re-export)
│
├── fast_strategy.py   # Optimized evaluation (numba, pre-computed matrices)
└── strategy.py        # DEPRECATED: Backward compatibility shim
```

## Data Flow

Here's how a typical game simulation works:

```
1. Load word list
   └─> data/corpus.py → data/word_list.py → WordList

2. Create player with scorer
   └─> scoring/strategies.py → InformationGainScorer
   └─> game/player.py → Player(scorer)

3. For each turn:
   a. Player chooses guess
      └─> player.choose_guess(state)
      └─> scorer.score(guess, word_list) for all candidates
      └─> Returns best scoring word
   
   b. Compute result
      └─> core/result.py → compute_result(guess, answer)
      └─> Returns "GGGGG", "NYGYN", etc.
   
   c. Filter word list
      └─> filtering/filter.py → filter_word_list(guess, result, word_list)
      └─> core/constraints.py → parse_response(guess, result)
      └─> Returns filtered WordList
   
   d. Update game state
      └─> game/engine.py → play_turn(state, guess, result, filtered_words)
      └─> Returns new GameState
   
   e. Repeat until solved
```

## Core Concepts

### 1. Word Lists (`WordList`)

A `WordList` is a dictionary mapping words to their frequency data:

```python
WordList = dict[str, tuple[float, frozenset[str]]]
# Example:
{
    "CRANE": (1.23e-5, frozenset({'C', 'R', 'A', 'N', 'E'})),
    "SLATE": (8.45e-6, frozenset({'S', 'L', 'A', 'T', 'E'})),
    # ... more words
}
```

Each entry contains:
- **Frequency**: Raw frequency from wordfreq (common words have higher values)
- **Character set**: Pre-computed set of letters (for fast filtering)

### 2. Results (G/Y/N Encoding)

Wordle gives feedback for each letter:
- **G** = Green: correct letter, correct position
- **Y** = Yellow: correct letter, wrong position  
- **N** = Grey: letter not in word

A result is a 5-character string like `"GGGGG"` (all correct) or `"NYGYN"` (mixed).

### 3. Information Gain (Entropy)

The core scoring strategy uses **information gain** (Shannon entropy). For a guess:

1. **Partition** remaining words into buckets by what result they'd produce
2. **Calculate entropy**: `sum(p * log2(1/p))` where `p` = probability of each bucket
3. Higher entropy = more information gained = better guess

The scorer weights by word frequency, so eliminating common words (likely answers) is prioritized.

### 4. Constraints

After a guess, constraints are extracted:
- **exact**: Letters at exact positions (greens)
- **present**: Letters that exist somewhere (yellows)
- **absent_at**: Letters at wrong positions (yellow positions)
- **absent**: Letters not in word (greys)

These are used to filter the word list efficiently.

## Understanding the Result Matrix

The **result matrix** is an optimization technique used in `fast_strategy.py` for large-scale evaluation. Here's what it does and why it's useful.

### What is the Result Matrix?

A result matrix is a pre-computed table that stores what result every guess would produce against every possible answer.

**Size**: If you have N words, the matrix is N×N.

**Values**: Each entry `matrix[i][j]` is an integer (0-242) representing the result when:
- Word `i` is the guess
- Word `j` is the answer

**Encoding**: The 243 possible results (3^5: G/Y/N for each of 5 positions) are encoded as integers 0-242.

### Example

With 3 words: `["CRANE", "SLATE", "TRACE"]`

The matrix might look like:

```
          CRANE  SLATE  TRACE
CRANE    [ 0  ,  100,   50  ]  ← if CRANE is guess, results for each answer
SLATE    [ 180,  0  ,   120 ]  ← if SLATE is guess...
TRACE    [ 80 ,  200,   0   ]  ← if TRACE is guess...
```

Where `0` = "GGGGG" (exact match), and other numbers encode mixed results.

### Why Use a Matrix?

**Problem**: During evaluation, you might simulate thousands of games. Each game requires computing results for many guess-answer pairs.

**Without matrix**: Every `compute_result(guess, answer)` call does:
- Loop through positions
- Mark greens, yellows, greys
- String operations

**With matrix**: Once pre-computed, lookups are just `matrix[guess_idx][answer_idx]` - instant!

**Tradeoff**: 
- **Memory**: O(N²) - can be large (10K words = 400MB)
- **Setup time**: O(N²) - takes time to compute initially
- **Lookup time**: O(1) - extremely fast

### When to Use FastEvaluator

Use `FastEvaluator` (which uses the result matrix) when:
- ✅ Evaluating many starting words (100s-1000s)
- ✅ Running parameter optimization (many evaluations)
- ✅ Batch testing against many answers

Use regular `Player`/`Scorer` when:
- ✅ Playing single games interactively
- ✅ Understanding/debugging the strategy
- ✅ Research/experimentation where clarity > speed

### How FastEvaluator Uses the Matrix

```python
# 1. Pre-compute matrix once (expensive, but done once)
matrix = build_result_matrix(words)  # N×N matrix

# 2. During evaluation (fast lookups)
for answer_idx in range(n_words):
    remaining_mask = np.ones(n_words, dtype=bool)
    
    for turn in range(max_guesses):
        guess_idx = choose_best_guess(remaining_mask)
        result = matrix[guess_idx][answer_idx]  # ← Instant lookup!
        
        # Filter: keep only words where matrix[guess_idx][word_idx] == result
        remaining_mask = remaining_mask & (matrix[guess_idx] == result)
```

The matrix allows filtering using vectorized numpy operations, which is much faster than Python loops.

## Module Details

### Core Module

**`core/result.py`**
- `compute_result(guess, answer) -> str`: Converts guess+answer into G/Y/N string
- Two-pass algorithm: first mark greens, then yellows
- Cached for performance

**`core/constraints.py`**
- `parse_response(guess, result) -> Constraints`: Extracts filterable constraints
- Handles edge cases (duplicate letters, confirmed vs unconfirmed)

### Data Module

**`data/corpus.py`**
- `load_raw_word_list() -> dict[str, float]`: Loads words from wordfreq
- Filters to 5-letter words, removes simple plurals

**`data/word_list.py`**
- `WordList` type alias
- `WordListConfig`: Configuration for creating word lists
- `create_word_list(config)`: Unified word list creation

**`data/frequency.py`**
- `FrequencyConfig`: Sigmoid transformation parameters (k, x0)
- `transform_word_list()`: Applies sigmoid to compress frequency range
- Inspection functions: `find_cutoff_word()`, `get_cutoff_stats()`, etc.

**`data/optimize.py`**
- `optimize_sigmoid_params()`: Finds optimal k/x0 parameters
- Uses FastEvaluator for speed
- Grid search or differential evolution

### Filtering Module

**`filtering/filter.py`**
- `filter_word_list(guess, result, word_list)`: Filters words based on constraints
- `ismatch(word, constraints)`: Checks if word satisfies constraints
- Uses pre-computed character sets for O(1) set operations

### Scoring Module

**`scoring/partition.py`**
- `partition_by_result(guess, word_list)`: Groups words by result they'd produce
- Returns dict: `{"GGGGG": ([words], freq_sum), "NYGYN": ([words], freq_sum), ...}`

**`scoring/base.py`**
- `Scorer` abstract base class
- `score(guess, word_list) -> float`: Score a single guess
- `rank_candidates(word_list, top_n) -> list[tuple[str, float]]`: Rank candidates

**`scoring/strategies.py`**
- `InformationGainScorer`: Uses entropy (higher = better)
- `ExpectedRemainingScorer`: Uses expected remaining words (lower = better, returns negative)

### Game Module

**`game/state.py`**
- `GameState`: Immutable game state
  - `remaining_words`: Current word list
  - `history`: List of (guess, result) tuples
  - `turn`: Current turn number
  
- `StrategyState`: Strategy analysis
  - `top_candidates`: Top guesses with scores
  - `remaining_count`: Number of remaining words
  - `last_guess`, `last_result`: Most recent move

**`game/engine.py`**
- `play_turn(state, guess, result, filtered_words) -> GameState`: Pure function, creates new state
- `is_solved(result) -> bool`: Checks if result is "GGGGG"

**`game/player.py`**
- `Player(scorer)`: Player with a scoring strategy
- `choose_guess(state) -> str`: Selects best guess using scorer
- `analyze_state(state, top_n) -> StrategyState`: Returns strategy analysis

### Evaluation Module

**`evaluation/simulator.py`**
- `simulate_game(answer, player, word_list, ...) -> GameResult`: Simulates a full game
- Returns `GameResult` with guesses, success, history, choices at each turn

**`evaluation/benchmark.py`**
- `evaluate_strategy(scorer, word_list, answers, ...) -> EvaluationResult`: Tests strategy against answer set
- Returns average guesses, failure rate, distribution

**`evaluation/optimized_evaluator.py`**
- Re-exports `FastEvaluator` from `fast_strategy.py`
- Use for large-scale evaluation

## Usage Examples

### Basic Game Simulation

```python
from wordle_helper.data import create_word_list, WordListConfig
from wordle_helper.scoring.strategies import InformationGainScorer
from wordle_helper.game.player import Player
from wordle_helper.evaluation.simulator import simulate_game

# Create word list
word_list = create_word_list(WordListConfig(max_words=10000))

# Create player
scorer = InformationGainScorer()
player = Player(scorer=scorer)

# Simulate a game
result = simulate_game(
    answer="CRANE",
    player=player,
    word_list=word_list,
)

print(f"Solved in {result.guesses} guesses: {result.success}")
```

### Inspecting Strategy State

```python
from wordle_helper.game.state import GameState
from wordle_helper.game.player import Player
from wordle_helper.scoring.strategies import InformationGainScorer

player = Player(scorer=InformationGainScorer())
state = GameState(remaining_words=word_list)

# See what the strategy is thinking
strategy_state = player.analyze_state(state, top_n=10)
print(f"Remaining words: {strategy_state.remaining_count}")
print("Top candidates:")
for word, score in strategy_state.top_candidates:
    print(f"  {word}: {score:.3f} bits")
```

### Inspecting Frequency Optimization

```python
from wordle_helper.data.frequency import FrequencyConfig, find_cutoff_word

# Check cutoff point
word_list = create_word_list(WordListConfig(max_words=10000))
cutoff_word, cutoff_freq = find_cutoff_word(word_list, k=10.0, x0=0.5)
print(f"Cutoff word: {cutoff_word} (transformed freq: {cutoff_freq})")

# Compare different parameters
from wordle_helper.data.frequency import compare_cutoffs
stats = compare_cutoffs(word_list, [(10.0, 0.5), (15.0, 0.5), (10.0, 0.6)])
for (k, x0), stat in stats.items():
    print(f"k={k}, x0={x0}: {stat['cutoff_word']} at cutoff")
```

### Using FastEvaluator for Batch Evaluation

```python
from wordle_helper.fast_strategy import FastEvaluator

# Create evaluator (builds result matrix internally)
evaluator = FastEvaluator(word_list, answer_indices=list(range(1000)))

# Evaluate a starting word
result = evaluator.evaluate_starting_word("SLATE")
print(f"Average: {result['average_score']:.2f} guesses")
print(f"Failure rate: {result['failure_rate']:.1%}")

# Rank all starting words (fast because matrix is pre-computed)
results = evaluator.rank_all_starting_words(parallel=True)
for i, r in enumerate(results[:10], 1):
    print(f"{i}. {r['starting_word']}: {r['average_score']:.3f}")
```

### Custom Scoring Strategy

```python
from wordle_helper.scoring.base import Scorer
from wordle_helper.data.word_list import WordList

class FrequencyScorer(Scorer):
    """Simple scorer that just uses word frequency."""
    
    def score(self, guess: str, word_list: WordList) -> float:
        if guess not in word_list:
            return 0.0
        freq, _ = word_list[guess]
        return freq  # Higher frequency = better score

# Use it
player = Player(scorer=FrequencyScorer())
guess = player.choose_guess(state)
```

## Key Design Decisions

1. **Immutable GameState**: State is never mutated, always returns new state. Makes inspection and debugging easier.

2. **Pluggable Scorers**: Easy to swap scoring strategies. Just change `Player(scorer=YourScorer())`.

3. **Separation of Concerns**: 
   - Core: Low-level primitives
   - Data: Word lists and transformations
   - Scoring: Strategy logic
   - Game: Orchestration
   - Evaluation: Testing/benchmarking

4. **Performance vs Clarity**: 
   - Default path: Clear, modular, easy to understand
   - Optimized path: FastEvaluator for batch operations

5. **Research-Friendly**: StrategyState, cutoff inspection, etc. make it easy to understand what's happening at each step.

