# cilpy/solver/decomposition/cpso.py
"""Static cooperative solver (CPSO) with a fixed grouping."""

from typing import Optional, Type

from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import GroupingStrategy, StaticEvenGrouping

class StandardCooperativeSolver(CooperativeDecompositionSolver):
    """Cooperative decomposition solver with a fixed grouping.

    Splits the variables into groups once at the start and never
    regroups them.
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
        """Initializes the static cooperative solver.

        Args:
            problem: The full problem to solve.
            name: The name of the solver.
            num_groups: The number of groups. Can be any number from 1 to
                the problem dimension.
            sub_solver_class: The solver class used for every group, for
                example PSO, GA or DE.
            sub_solver_params: The parameters passed to every sub-solver.
            grouping_strategy: Decides how the variables are put into groups.
                Defaults to StaticEvenGrouping.
            constraint_handler: Optional constraint handler for the coordinator.
        """
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
        """Does nothing, because the groups never change."""
        pass
