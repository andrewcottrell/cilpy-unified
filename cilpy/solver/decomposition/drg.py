# cilpy/solver/decomposition/drg.py
from typing import Optional, Type
from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import GroupingStrategy, RandomGrouping
from cilpy.solver.decomposition.trigger import RegroupTrigger, FixedIntervalTrigger

class DynamicRandomGroupingSolver(CooperativeDecompositionSolver):
    """Cooperative solver that randomly regroups the dimensions when the trigger fires."""

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
        """Randomly regroups the dimensions when the trigger fires, keeping the same number of groups."""

        state = self._build_state_dict()

        if self._regroup_trigger.should_trigger(state):

            new_groups = self._grouping_strategy.group(
                self.problem.dimension, len(self._groups))

            self._perform_regroup(new_groups)
            self._regroup_trigger.on_triggered(state)
