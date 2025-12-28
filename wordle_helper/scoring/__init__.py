"""Scoring module for evaluating word guesses."""

from wordle_helper.scoring.base import Scorer
from wordle_helper.scoring.strategies import (
    InformationGainScorer,
    ExpectedRemainingScorer,
)
from wordle_helper.scoring.partition import partition_by_result

__all__ = [
    "Scorer",
    "InformationGainScorer",
    "ExpectedRemainingScorer",
    "partition_by_result",
]

