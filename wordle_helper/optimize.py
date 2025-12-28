"""
Parameter optimization for sigmoid frequency transformation.

Finds optimal sigmoid parameters (k, x0) that minimize average guesses
against a provided answer list.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Callable
from tqdm import tqdm

from wordle_helper.frequency import transform_word_list
from wordle_helper.fast_strategy import FastEvaluator, build_result_matrix


def evaluate_params(
    k: float,
    x0: float,
    word_list: dict,
    answer_words: list[str],
    max_guesses: int = 6,
    fail_penalty: float = 10.0,
    result_matrix: np.ndarray | None = None,
) -> float:
    """
    Evaluate sigmoid parameters by running full evaluation.

    Args:
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        word_list: Base word list (untransformed)
        answer_words: List of answer words to evaluate against
        max_guesses: Maximum guesses per game
        fail_penalty: Penalty for failed games
        result_matrix: Pre-computed result matrix (avoids rebuilding)

    Returns:
        Average number of guesses (lower is better)
    """
    # Apply sigmoid transform
    transformed = transform_word_list(word_list, k=k, x0=x0)

    # Map answer words to indices
    words = list(transformed.keys())
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Filter to answers that exist in word list
    answer_indices = []
    for word in answer_words:
        word_upper = word.upper()
        if word_upper in word_to_idx:
            answer_indices.append(word_to_idx[word_upper])

    if not answer_indices:
        raise ValueError("No answer words found in word list")

    # Build evaluator with transformed frequencies (reuse result matrix)
    evaluator = FastEvaluator(
        transformed,
        answer_indices=answer_indices,
        result_matrix=result_matrix,
    )

    # Find best starting word once (same for all answers)
    starting_idx = evaluator.best_guess_idx(
        np.ones(evaluator.n_words, dtype=bool),
        use_cache=True
    )

    # Evaluate against all answer words
    total_score = 0.0
    for answer_idx in answer_indices:
        guesses = evaluator.play_game(answer_idx, starting_idx, max_guesses)
        if guesses > max_guesses:
            total_score += fail_penalty
        else:
            total_score += guesses

    return total_score / len(answer_indices)


def _objective(
    params: np.ndarray,
    word_list: dict,
    answer_words: list[str],
    result_matrix: np.ndarray,
) -> float:
    """Objective function for scipy optimizer."""
    k, x0 = params
    try:
        return evaluate_params(k, x0, word_list, answer_words, result_matrix=result_matrix)
    except Exception as e:
        print(f"Error with k={k:.2f}, x0={x0:.2f}: {e}")
        return 100.0  # High penalty for errors


def optimize_sigmoid_params(
    word_list: dict,
    answer_words: list[str],
    k_range: tuple[float, float] = (1.0, 50.0),
    x0_range: tuple[float, float] = (0.2, 0.8),
    method: str = "grid",
    n_grid: int = 10,
    verbose: bool = True,
    result_matrix: np.ndarray | None = None,
) -> dict:
    """
    Find optimal sigmoid parameters to minimize average guesses.

    Args:
        word_list: Base word list (untransformed)
        answer_words: List of answer words to optimize against
        k_range: Range for k parameter (steepness)
        x0_range: Range for x0 parameter (midpoint)
        method: Optimization method - "grid", "differential_evolution", or "nelder-mead"
        n_grid: Grid resolution for grid search
        verbose: Whether to show progress
        result_matrix: Pre-computed result matrix (builds once if not provided)

    Returns:
        Dict with:
            - best_k: Optimal steepness
            - best_x0: Optimal midpoint
            - best_avg_guesses: Best average guesses achieved
            - trials: List of (k, x0, avg_guesses) for all trials
            - baseline_avg_guesses: Average guesses with no transform (k=0 equivalent)
    """
    # Pre-compute result matrix once (this is the expensive part)
    if result_matrix is None:
        words = list(word_list.keys())
        if verbose:
            print(f"Pre-computing result matrix for {len(words)} words...")
        result_matrix = build_result_matrix(words, show_progress=verbose)

    trials = []

    if method == "grid":
        # Grid search
        k_values = np.linspace(k_range[0], k_range[1], n_grid)
        x0_values = np.linspace(x0_range[0], x0_range[1], n_grid)

        best_score = float('inf')
        best_k, best_x0 = k_values[0], x0_values[0]

        total = len(k_values) * len(x0_values)
        iterator = tqdm(total=total, desc="Grid search") if verbose else None

        for k in k_values:
            for x0 in x0_values:
                score = evaluate_params(k, x0, word_list, answer_words, result_matrix=result_matrix)
                trials.append((k, x0, score))

                if score < best_score:
                    best_score = score
                    best_k, best_x0 = k, x0
                    if verbose:
                        print(f"\nNew best: k={k:.1f}, x0={x0:.2f}, avg={score:.4f}")

                if iterator:
                    iterator.update(1)

        if iterator:
            iterator.close()

    elif method == "differential_evolution":
        # Differential evolution (global optimizer)
        bounds = [k_range, x0_range]

        if verbose:
            print("Running differential evolution optimization...")

        result = differential_evolution(
            lambda p: _objective(p, word_list, answer_words, result_matrix),
            bounds,
            maxiter=50,
            seed=42,
            disp=verbose,
            workers=1,  # Avoid nested parallelism
        )

        best_k, best_x0 = result.x
        best_score = result.fun
        trials.append((best_k, best_x0, best_score))

    elif method == "nelder-mead":
        # Local optimization from center of range
        x0_init = [(k_range[0] + k_range[1]) / 2, (x0_range[0] + x0_range[1]) / 2]

        if verbose:
            print("Running Nelder-Mead optimization...")

        result = minimize(
            lambda p: _objective(p, word_list, answer_words, result_matrix),
            x0_init,
            method='Nelder-Mead',
            options={'maxiter': 100, 'disp': verbose},
        )

        best_k, best_x0 = result.x
        best_score = result.fun
        trials.append((best_k, best_x0, best_score))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute baseline (very low k = almost uniform weighting)
    if verbose:
        print("\nComputing baseline (k=1, x0=0.5)...")
    baseline_score = evaluate_params(1.0, 0.5, word_list, answer_words, result_matrix=result_matrix)

    return {
        "best_k": best_k,
        "best_x0": best_x0,
        "best_avg_guesses": best_score,
        "baseline_avg_guesses": baseline_score,
        "improvement": baseline_score - best_score,
        "trials": trials,
        "result_matrix": result_matrix,  # Return for reuse
    }


def quick_optimize(
    word_list: dict,
    answer_words: list[str],
    verbose: bool = True,
) -> dict:
    """
    Quick optimization with sensible defaults.

    Uses a coarse grid search followed by local refinement.
    Pre-computes result matrix once and reuses across both phases.

    Args:
        word_list: Base word list
        answer_words: Answer words to optimize against
        verbose: Show progress

    Returns:
        Optimization results dict
    """
    # Pre-compute result matrix once for both phases
    words = list(word_list.keys())
    if verbose:
        print(f"Pre-computing result matrix for {len(words)} words...")
    result_matrix = build_result_matrix(words, show_progress=verbose)

    # Coarse grid search
    if verbose:
        print("\nPhase 1: Coarse grid search...")

    coarse_result = optimize_sigmoid_params(
        word_list,
        answer_words,
        k_range=(2, 40),
        x0_range=(0.3, 0.7),
        method="grid",
        n_grid=5,
        verbose=verbose,
        result_matrix=result_matrix,
    )

    # Fine grid around best
    if verbose:
        print(f"\nPhase 2: Fine search around k={coarse_result['best_k']:.1f}, x0={coarse_result['best_x0']:.2f}...")

    k_best = coarse_result['best_k']
    x0_best = coarse_result['best_x0']

    fine_result = optimize_sigmoid_params(
        word_list,
        answer_words,
        k_range=(max(1, k_best - 10), min(50, k_best + 10)),
        x0_range=(max(0.1, x0_best - 0.15), min(0.9, x0_best + 0.15)),
        method="grid",
        n_grid=7,
        verbose=verbose,
        result_matrix=result_matrix,
    )

    # Combine trials
    fine_result['trials'] = coarse_result['trials'] + fine_result['trials']
    fine_result['baseline_avg_guesses'] = coarse_result['baseline_avg_guesses']

    return fine_result


def load_answer_list(filepath: str) -> list[str]:
    """
    Load answer words from a file.

    Supports formats:
    - One word per line: "CRANE"
    - Date-prefixed: "2021-06-19 CIGAR"

    Args:
        filepath: Path to answer list file

    Returns:
        List of answer words (uppercase)
    """
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle "DATE WORD" format
            parts = line.split()
            word = parts[-1].upper()  # Take last part as word
            if len(word) == 5 and word.isalpha():
                words.append(word)
    return words
