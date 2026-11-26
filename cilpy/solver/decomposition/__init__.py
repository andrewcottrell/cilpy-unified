# cilpy/solver/decomposition/__init__.py
"""Cooperative decomposition solvers for large-scale optimization.

The package holds the four cooperative solvers (static cooperative, dynamic
random grouping, decomposition cooperative and merging cooperative) and the
groupings, splitters, mergers and triggers they are built from.
"""

from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem
from cilpy.solver.decomposition.grouping import (
    GroupingStrategy, StaticEvenGrouping, RandomGrouping,
    RandomGroupSizeGrouping, GroupSplitter, EvenSplitter,
    RandomSplitter, GroupMerger, EvenMerger, RandomMerger)
from cilpy.solver.decomposition.trigger import (
    RegroupTrigger, FixedIntervalTrigger, EvaluationIntervalTrigger,
    StagnationTrigger, NeverTrigger, restructure_interval, restructure_trigger)
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.cpso import StandardCooperativeSolver
from cilpy.solver.decomposition.drg import DynamicRandomGroupingSolver
from cilpy.solver.decomposition.dcpso import DecompositionCooperativeSolver
from cilpy.solver.decomposition.mcpso import MergingCooperativeSolver

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
    "GroupMerger",
    "EvenMerger",
    "RandomMerger",
    "RegroupTrigger",
    "FixedIntervalTrigger",
    "EvaluationIntervalTrigger",
    "StagnationTrigger",
    "NeverTrigger",
    "restructure_interval",
    "restructure_trigger",
    "CooperativeDecompositionSolver",
    "StandardCooperativeSolver",
    "DynamicRandomGroupingSolver",
    "DecompositionCooperativeSolver",
    "MergingCooperativeSolver",
]
