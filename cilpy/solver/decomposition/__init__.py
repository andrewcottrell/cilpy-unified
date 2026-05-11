# cilpy/solver/decomposition/__init__.py
"""Decomposition-based solvers for large-scale optimization.
"""

from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem
from cilpy.solver.decomposition.grouping import GroupingStrategy, StaticEvenGrouping, RandomGrouping
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver

__all__ = [
    "ContextVector",
    "SubSpaceProblem",
    "GroupingStrategy",
    "StaticEvenGrouping",
    "RandomGrouping",
    "CooperativeDecompositionSolver",
    "StandardCooperativeSolver",
]
