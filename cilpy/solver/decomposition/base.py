# cilpy/solver/decomposition/base.py
"""Abstract coordinator for cooperative decomposition solvers."""

from abc import abstractmethod
from typing import List, Optional, Tuple, Type

from cilpy.problem import Evaluation, Problem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.grouping import GroupingStrategy
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem

class CooperativeDecompositionSolver(Solver):
    """Abstract base that partitions a D-dim problem into sub-problems.
    """

    def __init__(
        self,
        problem: Problem,
        name: str,
        num_groups: int,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        grouping_strategy: GroupingStrategy,
        constraint_handler: Optional[ConstraintHandler] = None,
        **kwargs,
    ):
        super().__init__(problem, name, constraint_handler)

        if not (1 <= num_groups <= problem.dimension):
            raise ValueError(
                f"num_groups must be between 1 and {problem.dimension}, not {num_groups}"
            )

        self._grouping_strategy = grouping_strategy
        self._groups: List[List[int]] = grouping_strategy.group(
            problem.dimension, num_groups
        )

        self._context = ContextVector(
            problem.dimension,
            self._groups,
            initial_vector=list(problem.bounds[0]),
        )

        self._sub_problems: List[SubSpaceProblem] = [
            SubSpaceProblem(problem, self._context, k, self._groups[k])
            for k in range(num_groups)
        ]

        self._sub_solvers: List[Solver] = self._instantiate_sub_solvers(
            sub_solver_class, sub_solver_params, num_groups
        )
        self._sub_solver_class = sub_solver_class
        self._sub_solver_params = sub_solver_params

        self._iteration: int = 0
        self._best_fitness_history: List[float] = []
        self._last_regroup_iteration: int = 0

        self._sync_context()

    def _instantiate_sub_solvers(
        self,
        sub_solver_class: Type[Solver],
        sub_solver_params: dict,
        num_groups: int,
    ) -> List[Solver]:
        solvers = []
        base_name = sub_solver_params.get("name", self.name)
        for k in range(num_groups):
            params = dict(sub_solver_params)
            params["name"] = f"{base_name}_sub{k}"
            solvers.append(
                sub_solver_class(problem=self._sub_problems[k], **params)
            )
        return solvers

    def _sync_context(self) -> None:
        """Pull each sub-solver's best partial solution into the context."""
        for k, sub_solver in enumerate(self._sub_solvers):
            best_sub_solution, _ = sub_solver.get_result()[0]
            self._context.update_group(k, best_sub_solution)

    def _get_global_best(self) -> Tuple[List[float], Evaluation]:
        """Build the full vector from context and evaluate on the original problem."""
        full_vector = self._context.get_full_vector()
        evaluation = self.problem.evaluate(full_vector)
        return (full_vector, evaluation)

    def step(self) -> None:
        self._iteration += 1

        for k, sub_solver in enumerate(self._sub_solvers):
            sub_solver.refresh_evaluations()
            sub_solver.step()
            best_sub_solution, _ = sub_solver.get_result()[0]
            self._context.update_group(k, best_sub_solution)

        current_fitness = self._get_global_best()[1].fitness
        self._best_fitness_history.append(current_fitness)

        self._post_step_hook()

    @abstractmethod
    def _post_step_hook(self) -> None:
        """Will implement this later"""

    def get_population(self) -> List[List[float]]:
        expanded = []

        for k, sub_solver in enumerate(self._sub_solvers):
            for sub_individual in sub_solver.get_population():
                full = self._context.get_full_vector()

                for local_idx, global_idx in enumerate(self._groups[k]):
                    full[global_idx] = sub_individual[local_idx]

                expanded.append(full)
        return expanded

    def get_population_evaluations(self) -> List[Evaluation]:

        evaluations = []
        for individual in self.get_population():
            evaluation = self.problem.evaluate(individual)
            evaluations.append(evaluation)
        return evaluations

    def get_result(self) -> List[Tuple[List[float], Evaluation]]:
        return [self._get_global_best()]