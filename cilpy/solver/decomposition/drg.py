# cilpy/solver/decomposition/drg.py
"""Dynamic random grouping solver (DRG)."""

from typing import Optional, Type
from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import GroupingStrategy, RandomGrouping
from cilpy.solver.decomposition.trigger import RegroupTrigger, FixedIntervalTrigger

class DynamicRandomGroupingSolver(CooperativeDecompositionSolver):
    """Cooperative solver that randomly regroups the dimensions when the trigger fires.

    Regrouping gives variables that interact a chance to end up in the same
    group. The search state is kept across a regroup, and a group's new best
    is only written into the context when it does not make the context worse.
    """

    def __init__(
        self,
        problem: Problem,
        name: str,
        num_groups: int,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        grouping_strategy: Optional[GroupingStrategy] = None,
        regroup_trigger: Optional[RegroupTrigger] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        """Initializes the dynamic random grouping solver.

        Args:
            problem: The full problem to solve.
            name: The name of the solver.
            num_groups: The number of groups. Ignored by RandomGroupSizeGrouping,
                which picks its own number of groups.
            sub_solver_class: The solver class used for every group, for
                example PSO, GA or DE.
            sub_solver_params: The parameters passed to every sub-solver.
            grouping_strategy: Decides how the variables are put into groups
                at the start and at every regroup. Defaults to RandomGrouping.
            regroup_trigger: Decides when to regroup. Defaults to
                FixedIntervalTrigger with an interval of 50 iterations.
            constraint_handler: Optional constraint handler for the coordinator.
        """
        super().__init__(
            problem=problem,
            name=name,
            num_groups=num_groups,
            sub_solver_class=sub_solver_class,
            sub_solver_params=sub_solver_params,
            grouping_strategy=grouping_strategy or RandomGrouping(),
            constraint_handler=constraint_handler,
            only_accept_improvements=True,
            **kwargs,
        )

        self._regroup_trigger = regroup_trigger or FixedIntervalTrigger(interval=50)
        self._regroup_trigger.reset()

    def _post_step_hook(self) -> None:
        """Regroups the dimensions with the grouping strategy when the trigger fires."""

        state = self._build_state_dict()

        if self._regroup_trigger.should_trigger(state):

            new_groups = self._grouping_strategy.group(
                self.problem.dimension, len(self._groups))

            self._perform_regroup(new_groups)
            self._regroup_trigger.on_triggered(state)
