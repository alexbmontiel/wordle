"""Evaluation module for testing and benchmarking strategies."""

from wordle_helper.evaluation.simulator import simulate_game, GameResult
from wordle_helper.evaluation.benchmark import evaluate_strategy, EvaluationResult

__all__ = [
    "simulate_game",
    "GameResult",
    "evaluate_strategy",
    "EvaluationResult",
]

