# cilpy/solver/decomposition/dcpso.py
"""Decomposition Cooperative solver (DCPSO)."""

from typing import Optional, Type
from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import (EvenSplitter, GroupingStrategy, GroupSplitter, StaticEvenGrouping)
from cilpy.solver.decomposition.trigger import (RegroupTrigger, restructure_trigger)

class DecompositionCooperativeSolver(CooperativeDecompositionSolver):
    """Cooperative solver that starts with one group and keeps splitting it."""

    def __init__(
        self,
        problem: Problem,
        name: str,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        num_groups: int = 1,
        grouping_strategy: Optional[GroupingStrategy] = None,
        group_splitter: Optional[GroupSplitter] = None,
        regroup_trigger: Optional[RegroupTrigger] = None,
        max_iterations: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):

        if regroup_trigger is None:
            if max_iterations is None and max_evaluations is None:
                raise ValueError(
                    "either regroup_trigger, max_iterations or max_evaluations must be given")

            if max_iterations is not None and max_evaluations is not None:
                raise ValueError("give either max_iterations or max_evaluations, not both")

        super().__init__(
            problem=problem,
            name=name,
            num_groups=num_groups,
            sub_solver_class=sub_solver_class,
            sub_solver_params=sub_solver_params,
            grouping_strategy=grouping_strategy or StaticEvenGrouping(),
            constraint_handler=constraint_handler,
            only_accept_improvements=True,
            **kwargs,
        )

        self._group_splitter = group_splitter or EvenSplitter(split_factor=2)

        if regroup_trigger is None:
            regroup_trigger = restructure_trigger(
                problem.dimension, self._group_splitter.split_factor,
                max_iterations=max_iterations, max_evaluations=max_evaluations)

        self._regroup_trigger = regroup_trigger
        self._regroup_trigger.reset()

    def _is_fully_decomposed(self) -> bool:
        """Returns True when every group has only one dimension."""

        for group in self._groups:
            if len(group) > 1:
                return False

        return True

    def _post_step_hook(self) -> None:
        """Splits all the groups when the trigger fires until every group has one dimension."""

        # Stop when all groups have only one variable
        if self._is_fully_decomposed():
            return

        state = self._build_state_dict()

        if self._regroup_trigger.should_trigger(state):

            new_groups = self._group_splitter.split(self._groups, state)

            self._perform_regroup(new_groups)
            self._give_new_velocities()
            self._regroup_trigger.on_triggered(state)
