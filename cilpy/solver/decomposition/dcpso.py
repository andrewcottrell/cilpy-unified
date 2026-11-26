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
    """Cooperative solver that starts with one group and keeps splitting it.

    It starts as one solver over all the variables. Every time the trigger
    fires, every group is split, until each group holds one variable. After
    a split, PSO particles get new random velocities and GA or DE
    individuals (except the first) get new random positions.
    """

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
        """Initializes the decomposition cooperative solver.

        Args:
            problem: The full problem to solve.
            name: The name of the solver.
            sub_solver_class: The solver class used for every group, for
                example PSO, GA or DE.
            sub_solver_params: The parameters passed to every sub-solver.
            num_groups: The number of groups to start with. Defaults to 1.
            grouping_strategy: Decides how the variables are put into the
                starting groups. Defaults to StaticEvenGrouping.
            group_splitter: Decides how groups are split. Defaults to
                EvenSplitter with a split factor of 2.
            regroup_trigger: Decides when to split. If not given, a trigger
                is built that spaces the splits evenly over the budget.
            max_iterations: The iteration budget, used to build the default trigger.
            max_evaluations: The evaluation budget, used to build the default trigger.
            constraint_handler: Optional constraint handler for the coordinator.

        Raises:
            ValueError: If no trigger is given and neither or both budgets are given.
        """
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
            self._give_new_positions()
            self._regroup_trigger.on_triggered(state)
