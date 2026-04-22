"""
Auto parameter tuner.

Two modes:
  - grid_search: exhaustive over param_grid (used by local path)
  - llm_guided:  Claude suggest next params based on previous results (used by llm path)
"""
import itertools
from typing import Any
import numpy as np

from agents.match_result import MatchResult


def grid_search(
    algo: Any,
    template: np.ndarray,
    scene: np.ndarray,
    max_combos: int = 20,
) -> list[MatchResult]:
    """
    Run algorithm over a subset of its param_grid.
    Returns results sorted by confidence descending.
    """
    grid = algo.param_grid
    if not grid:
        return [algo.timed_run(template, scene)]

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))

    # Limit combinations
    if len(combos) > max_combos:
        # Sample evenly
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            r = algo.timed_run(template, scene, **params)
            results.append(r)
        except Exception as e:
            pass  # skip bad param combos

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


def best_result(results: list[MatchResult]) -> MatchResult | None:
    if not results:
        return None
    return max(results, key=lambda r: r.confidence)
