# cilpy/solver/decomposition/mcpso.py
from typing import Optional, Type
from cilpy.problem import Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.base import CooperativeDecompositionSolver
from cilpy.solver.decomposition.grouping import (EvenMerger, GroupingStrategy, GroupMerger, StaticEvenGrouping)
from cilpy.solver.decomposition.trigger import (FixedIntervalTrigger, RegroupTrigger, restructure_interval)

class MergingCooperativeSolver(CooperativeDecompositionSolver):
    """MCPSO solver"""

    def __init__(
        self,
        problem: Problem,
        name: str,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        num_groups: Optional[int] = None,
        grouping_strategy: Optional[GroupingStrategy] = None,
        group_merger: Optional[GroupMerger] = None,
        regroup_trigger: Optional[RegroupTrigger] = None,
        max_iterations: Optional[int] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        if regroup_trigger is None and max_iterations is None:
            raise ValueError("either regroup_trigger or max_iterations must be give")

        if num_groups is None:
            num_groups = problem.dimension

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

        self._group_merger = group_merger or EvenMerger(merge_factor=2)

        if regroup_trigger is None:
            interval = restructure_interval(
                max_iterations, problem.dimension, self._group_merger.merge_factor)
            regroup_trigger = FixedIntervalTrigger(interval=interval)
        self._regroup_trigger = regroup_trigger
        self._regroup_trigger.reset()

    def _post_step_hook(self) -> None:

        if len(self._groups) == 1:
            
            return

        state = self._build_state_dict()

        if self._regroup_trigger.should_trigger(state):

            new_groups = self._group_merger.merge(self._groups, state)

            self._perform_regroup(new_groups) 
            self._give_new_velocities()
            self._regroup_trigger.on_triggered(state)
