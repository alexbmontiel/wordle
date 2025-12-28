"""Optimized evaluation using pre-computed matrices.

This module re-exports FastEvaluator and build_result_matrix from fast_strategy.py
for use in optimized evaluation workflows.

Note: fast_strategy.py remains a separate module because:
- It's a large, self-contained optimization module (~500+ lines)
- It uses numba JIT compilation which requires careful structure
- It's actively used by optimize.py and existing notebooks
- Keeping it separate maintains clear separation between optimization code and
  the main modular game logic

For general use, prefer the evaluation.benchmark module which uses the
standard Player/Scorer interfaces.
"""

from wordle_helper.fast_strategy import FastEvaluator, build_result_matrix

__all__ = ["FastEvaluator", "build_result_matrix"]

