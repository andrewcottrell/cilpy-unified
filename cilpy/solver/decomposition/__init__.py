# cilpy/solver/decomposition/__init__.py
"""Decomposition-based solvers for large-scale optimization.
"""

from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem
from cilpy.solver.decomposition.grouping import (
    GroupingStrategy, StaticEvenGrouping, RandomGrouping,
    RandomGroupSizeGrouping, GroupSplitter, EvenSplitter,
    RandomSplitter)
from cilpy.solver.decomposition.trigger import (
    RegroupTrigger, FixedIntervalTrigger, StagnationTrigger,
    NeverTrigger, restructure_interval)
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.dcpso import DecompositionCooperativeSolver

__all__ = [
    "ContextVector",
    "SubSpaceProblem",
    "GroupingStrategy",
    "StaticEvenGrouping",
    "RandomGrouping",
    "RandomGroupSizeGrouping",
    "GroupSplitter",
    "EvenSplitter",
    "RandomSplitter",
    "RegroupTrigger",
    "FixedIntervalTrigger",
    "StagnationTrigger",
    "NeverTrigger",
    "restructure_interval",
    "CooperativeDecompositionSolver",
    "StandardCooperativeSolver",
    "DynamicRandomGroupingSolver",
    "DecompositionCooperativeSolver",
]
