"""
Frequency transformation and visualization tools.

Provides sigmoid transformation to adjust word frequency weighting,
plus visualization tools to understand where cutoffs occur.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.axes import Axes
from matplotlib.figure import Figure
try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wordle_helper.data.word_list import WordList
else:
    # Runtime type alias to avoid circular import
    WordList = dict[str, tuple[float, frozenset[str]]]


@dataclass
class FrequencyConfig:
    """
    Configuration for frequency transformation.
    
    Attributes:
        k: Sigmoid steepness parameter (higher = sharper cutoff)
        x0: Sigmoid midpoint parameter (0-1, where sigmoid = 0.5)
    """
    k: float = 10.0
    x0: float = 0.5


def sigmoid_transform(
    freq: float | np.ndarray,
    k: float = 10.0,
    x0: float = 0.5,
    freq_min: float | None = None,
    freq_max: float | None = None,
) -> float | np.ndarray:
    """
    Apply sigmoid transformation to frequency value(s).

    The transformation:
    1. Normalizes frequencies to [0, 1] using log scale
    2. Applies sigmoid: 1 / (1 + exp(-k * (x - x0)))

    Args:
        freq: Raw frequency value(s)
        k: Steepness parameter. Low (~1-5) = gradual compression,
           High (~20-50) = sharp cutoff
        x0: Midpoint in normalized space (0-1). Where sigmoid = 0.5
        freq_min: Min frequency for normalization (required)
        freq_max: Max frequency for normalization (required)

    Returns:
        Transformed frequency value(s) in range (0, 1)
    """
    if freq_min is None or freq_max is None:
        raise ValueError("freq_min and freq_max must be provided")

    # Normalize to [0, 1] using log scale
    log_freq = np.log10(freq)
    log_min = np.log10(freq_min)
    log_max = np.log10(freq_max)

    # Handle edge case where all frequencies are the same
    if log_max == log_min:
        return np.ones_like(freq) if isinstance(freq, np.ndarray) else 1.0

    normalized = (log_freq - log_min) / (log_max - log_min)

    # Sigmoid transformation
    return 1 / (1 + np.exp(-k * (normalized - x0)))


def get_freq_bounds(word_list: WordList) -> tuple[float, float]:
    """Get min and max frequencies from word list."""
    freqs = [f for f, _ in word_list.values()]
    return min(freqs), max(freqs)


def transform_word_list(
    word_list: WordList,
    k: float = 10.0,
    x0: float = 0.5,
) -> WordList:
    """
    Apply sigmoid transform to all frequencies in word list.

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter

    Returns:
        New word list with transformed frequencies
    """
    freq_min, freq_max = get_freq_bounds(word_list)

    transformed = {}
    for word, (freq, charset) in word_list.items():
        new_freq = sigmoid_transform(freq, k, x0, freq_min, freq_max)
        transformed[word] = (new_freq, charset)

    return transformed


def find_cutoff_word(word_list: WordList, k: float, x0: float) -> tuple[str, float]:
    """
    Find the word at the sigmoid cutoff point (transformed freq ≈ 0.5).
    
    Args:
        word_list: Word list to search
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        
    Returns:
        Tuple of (word, transformed_frequency) at cutoff point
    """
    freq_min, freq_max = get_freq_bounds(word_list)
    threshold = 0.5
    
    # Find word closest to threshold
    best_word = None
    best_dist = float('inf')
    best_freq = 0.0
    
    for word, (freq, _) in word_list.items():
        transformed = sigmoid_transform(freq, k, x0, freq_min, freq_max)
        dist = abs(transformed - threshold)
        if dist < best_dist:
            best_dist = dist
            best_word = word
            best_freq = transformed
    
    return best_word, best_freq


def get_words_near_cutoff(
    word_list: WordList,
    k: float,
    x0: float,
    n: int,
) -> list[tuple[str, float, float]]:
    """
    Get words near the cutoff point with raw and transformed frequencies.
    
    Args:
        word_list: Word list to search
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        n: Number of words to return (centered on cutoff)
        
    Returns:
        List of (word, raw_frequency, transformed_frequency) tuples,
        sorted by distance from cutoff
    """
    freq_min, freq_max = get_freq_bounds(word_list)
    threshold = 0.5
    
    word_weights = []
    for word, (freq, _) in word_list.items():
        transformed = sigmoid_transform(freq, k, x0, freq_min, freq_max)
        dist = abs(transformed - threshold)
        word_weights.append((dist, word, freq, transformed))
    
    # Sort by distance from threshold
    word_weights.sort()
    
    return [(word, raw_freq, transformed) for _, word, raw_freq, transformed in word_weights[:n]]


def get_cutoff_stats(word_list: WordList, k: float, x0: float) -> dict:
    """
    Get summary statistics about the cutoff.
    
    Args:
        word_list: Word list to analyze
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        
    Returns:
        Dictionary with stats:
        - n_above: Number of words above cutoff (>= 0.5)
        - n_below: Number of words below cutoff (< 0.5)
        - cutoff_word: Word at cutoff point
        - cutoff_raw_freq: Raw frequency of cutoff word
        - cutoff_transformed: Transformed frequency of cutoff word
    """
    freq_min, freq_max = get_freq_bounds(word_list)
    threshold = 0.5
    
    n_above = 0
    n_below = 0
    cutoff_word, cutoff_freq = find_cutoff_word(word_list, k, x0)
    cutoff_raw_freq = word_list[cutoff_word][0]
    
    for word, (freq, _) in word_list.items():
        transformed = sigmoid_transform(freq, k, x0, freq_min, freq_max)
        if transformed >= threshold:
            n_above += 1
        else:
            n_below += 1
    
    return {
        "n_above": n_above,
        "n_below": n_below,
        "cutoff_word": cutoff_word,
        "cutoff_raw_freq": cutoff_raw_freq,
        "cutoff_transformed": cutoff_freq,
    }


def find_params_for_cutoff_word(
    word_list: WordList,
    target_word: str,
    k_range: tuple[float, float] = (1.0, 50.0),
    x0_range: tuple[float, float] = (0.2, 0.8),
) -> tuple[float, float]:
    """
    Find k/x0 parameters that place a specific word at the cutoff (transformed freq ≈ 0.5).
    
    Uses optimization to find parameters that minimize distance from 0.5 for the target word.
    
    Args:
        word_list: Word list containing the target word
        target_word: Word to place at cutoff (must be in word_list)
        k_range: Range to search for k parameter
        x0_range: Range to search for x0 parameter
        
    Returns:
        Tuple of (k, x0) that places target_word at cutoff
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for find_params_for_cutoff_word. Install with: pip install scipy")
    
    if target_word not in word_list:
        raise ValueError(f"Target word '{target_word}' not in word list")
    
    target_raw_freq = word_list[target_word][0]
    freq_min, freq_max = get_freq_bounds(word_list)
    threshold = 0.5
    
    # Objective: minimize distance from threshold
    def objective(x0):
        transformed = sigmoid_transform(target_raw_freq, k=10.0, x0=x0, freq_min=freq_min, freq_max=freq_max)
        return abs(transformed - threshold)
    
    # First optimize x0 with fixed k=10.0
    result_x0 = minimize_scalar(objective, bounds=x0_range, method='bounded')
    best_x0 = result_x0.x
    
    # Then optimize k with fixed x0
    def objective_k(k):
        transformed = sigmoid_transform(target_raw_freq, k=k, x0=best_x0, freq_min=freq_min, freq_max=freq_max)
        return abs(transformed - threshold)
    
    result_k = minimize_scalar(objective_k, bounds=k_range, method='bounded')
    best_k = result_k.x
    
    return (best_k, best_x0)


def compare_cutoffs(
    word_list: WordList,
    params_list: list[tuple[float, float]],
) -> dict:
    """
    Compare multiple parameter sets and return statistics.
    
    Args:
        word_list: Word list to analyze
        params_list: List of (k, x0) tuples to compare
        
    Returns:
        Dictionary mapping (k, x0) -> cutoff_stats
    """
    results = {}
    for k, x0 in params_list:
        results[(k, x0)] = get_cutoff_stats(word_list, k, x0)
    return results


def find_cutoff_words(
    word_list: WordList,
    k: float = 10.0,
    x0: float = 0.5,
    n_words: int = 20,
    threshold: float = 0.5,
) -> list[tuple[str, float, float]]:
    """
    Find words near the sigmoid cutoff point.

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        n_words: Number of words to return (centered on threshold)
        threshold: Weight threshold to center on (default 0.5)

    Returns:
        List of (word, raw_frequency, transformed_weight) tuples,
        sorted by distance from threshold
    """
    return get_words_near_cutoff(word_list, k, x0, n_words)


def get_percentile_words(
    word_list: WordList,
    percentiles: list[int] = [10, 25, 50, 75, 90],
) -> dict[int, tuple[str, float]]:
    """
    Get representative words at frequency percentiles.

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        percentiles: List of percentiles to sample

    Returns:
        Dict mapping percentile -> (word, frequency)
    """
    words_sorted = sorted(
        word_list.items(),
        key=lambda x: x[1][0],
        reverse=True
    )
    n = len(words_sorted)

    result = {}
    for p in percentiles:
        idx = int((100 - p) / 100 * n)  # High freq = low index
        idx = min(idx, n - 1)
        word, (freq, _) = words_sorted[idx]
        result[p] = (word, freq)

    return result


def plot_frequency_distribution(
    word_list: WordList,
    ax: Axes | None = None,
    title: str = "Word Frequency Distribution",
) -> Axes:
    """
    Plot raw frequency distribution (log scale).

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    freqs = [f for f, _ in word_list.values()]
    indices = range(len(freqs))

    ax.semilogy(indices, sorted(freqs, reverse=True))
    ax.set_xlabel("Word Rank")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add percentile annotations
    percentile_words = get_percentile_words(word_list)
    for p, (word, freq) in percentile_words.items():
        idx = int((100 - p) / 100 * len(freqs))
        ax.annotate(
            f"{word}\n({p}%)",
            xy=(idx, freq),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5),
        )

    return ax


def plot_frequency_transform(
    word_list: WordList,
    k: float = 10.0,
    x0: float = 0.5,
    figsize: tuple = (12, 5),
    show_cutoff_words: bool = True,
) -> Figure:
    """
    Visualize raw vs transformed frequencies with cutoff annotation.

    Creates a two-panel figure:
    - Left: Raw frequency distribution with sigmoid overlay
    - Right: Transformed weights distribution

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        k: Sigmoid steepness parameter
        x0: Sigmoid midpoint parameter
        figsize: Figure size
        show_cutoff_words: Whether to annotate words near cutoff

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get frequency data
    words_sorted = sorted(word_list.items(), key=lambda x: x[1][0], reverse=True)
    words = [w for w, _ in words_sorted]
    freqs = np.array([f for _, (f, _) in words_sorted])
    freq_min, freq_max = freqs.min(), freqs.max()

    # Compute transformed weights
    weights = sigmoid_transform(freqs, k, x0, freq_min, freq_max)

    # Left plot: Raw frequencies with sigmoid curve overlay
    ax1.semilogy(range(len(freqs)), freqs, 'b-', alpha=0.7, label='Raw frequency')
    ax1.set_xlabel("Word Rank")
    ax1.set_ylabel("Frequency (log scale)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Add sigmoid curve on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(freqs)), weights, 'r-', alpha=0.7, label='Sigmoid weight')
    ax1_twin.set_ylabel("Transformed Weight", color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1_twin.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Cutoff (0.5)')

    ax1.set_title(f"Frequency vs Sigmoid Transform (k={k}, x0={x0})")
    ax1.grid(True, alpha=0.3)

    # Find and mark the cutoff point
    cutoff_idx = np.argmin(np.abs(weights - 0.5))
    ax1.axvline(x=cutoff_idx, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(
        f"Cutoff: {words[cutoff_idx]}\n(rank {cutoff_idx})",
        xy=(cutoff_idx, freqs[cutoff_idx]),
        xytext=(20, 20),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Right plot: Transformed weights histogram
    ax2.hist(weights, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0.5, color='r', linestyle='--', label='Cutoff (0.5)')
    ax2.set_xlabel("Transformed Weight")
    ax2.set_ylabel("Number of Words")
    ax2.set_title("Distribution of Transformed Weights")
    ax2.legend()

    # Add summary stats
    n_above = np.sum(weights >= 0.5)
    n_below = np.sum(weights < 0.5)
    ax2.text(
        0.95, 0.95,
        f"Above cutoff: {n_above}\nBelow cutoff: {n_below}",
        transform=ax2.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout()
    return fig


def plot_sigmoid_comparison(
    word_list: WordList,
    params_list: list[tuple[float, float]],
    figsize: tuple = (10, 6),
) -> Figure:
    """
    Compare different sigmoid parameters on same plot.

    Args:
        word_list: Dict mapping word -> (frequency, charset)
        params_list: List of (k, x0) tuples to compare
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    words_sorted = sorted(word_list.items(), key=lambda x: x[1][0], reverse=True)
    freqs = np.array([f for _, (f, _) in words_sorted])
    freq_min, freq_max = freqs.min(), freqs.max()

    for k, x0 in params_list:
        weights = sigmoid_transform(freqs, k, x0, freq_min, freq_max)
        ax.plot(range(len(freqs)), weights, label=f'k={k}, x0={x0}')

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Word Rank")
    ax.set_ylabel("Transformed Weight")
    ax.set_title("Sigmoid Transform Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

