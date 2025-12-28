"""Optimized scoring implementations using numba (optional)."""

"""
This module is a placeholder for future numba-optimized scorer implementations.

Note: The current optimization path uses FastEvaluator from fast_strategy.py,
which is re-exported via evaluation.optimized_evaluator. FastEvaluator uses
pre-computed result matrices and numba-optimized functions for large-scale
evaluation, but doesn't implement the Scorer interface.

If you need optimized scoring that implements the Scorer interface, you can:
1. Use the standard InformationGainScorer (pure Python, clear and debuggable)
2. Use FastEvaluator for batch evaluation (see evaluation.optimized_evaluator)

Future work could add OptimizedInformationGainScorer here that uses numba
to speed up the partition_by_result and entropy calculations while still
implementing the Scorer interface.
"""

try:
    # Try importing numba - if not available, optimized scorers won't work
    import numba  # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

__all__ = []

