"""Optimized scoring implementations using numba (optional)."""

# TODO: Consolidate fast_strategy.py into this module
# For now, this is a placeholder that will import from fast_strategy
# Eventually this will contain OptimizedInformationGainScorer, etc.

try:
    # Try importing numba - if not available, optimized scorers won't work
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

__all__ = []

