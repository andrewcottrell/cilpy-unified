# cilpy/solver/decomposition/cpso.py
"""Standard Cooperative PSO (CPSO-S_k) with static grouping."""

from typing import Optional, Type

from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import GroupingStrategy, StaticEvenGrouping

class StandardCooperativeSolver(CooperativeDecompositionSolver):
    """Cooperative decomposition solver with fixed static grouping.

    Partitions the search space once at initialisation and never
    regroups.
    """

    def __init__(
        self,
        problem: Problem,
        name: str,
        num_groups: int,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        grouping_strategy: Optional[GroupingStrategy] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            name=name,
            num_groups=num_groups,
            sub_solver_class=sub_solver_class,
            sub_solver_params=sub_solver_params,
            grouping_strategy=grouping_strategy or StaticEvenGrouping(),
            constraint_handler=constraint_handler,
            **kwargs,
        )

    def _post_step_hook(self) -> None:
        pass
