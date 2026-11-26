# cilpy/solver/decomposition/base.py
"""Abstract coordinator for cooperative decomposition solvers."""

import random
from abc import abstractmethod
from typing import List, Optional, Tuple, Type

from cilpy.problem import Evaluation, Problem
from cilpy.problem.counting import CountingProblem
from cilpy.solver import Solver
from cilpy.solver.chm import ConstraintHandler
from cilpy.solver.decomposition.context import ContextVector
from cilpy.solver.decomposition.grouping import GroupingStrategy
from cilpy.solver.decomposition.sub_problem import SubSpaceProblem

PER_DIMENSION_STATE_ATTRIBUTES = ("population", "velocities", "pbest_positions")

class PlaceholderProblem(Problem):
    """A placeholder sub-problem that never evaluates the real function.

    New sub-solvers are built on it during a regroup, so building them
    costs no fitness evaluations. Their state is then copied back in from
    the old sub-solvers.
    """

    def __init__(self, sub_problem: SubSpaceProblem) -> None:
        """Initializes the placeholder.

        Args:
            sub_problem: The real sub-problem. Its dimension, bounds and
                name are copied so the sub-solver is built with the right shape.
        """
        super().__init__(sub_problem.dimension, sub_problem.bounds, sub_problem.name)
        self._sub_problem = sub_problem

    def evaluate(self, solution) -> Evaluation:
        """Returns an infinite fitness without using an evaluation."""
        return Evaluation(fitness=float("inf"))

    def is_dynamic(self) -> Tuple[bool, bool]:
        """Returns whether the real sub-problem is dynamic."""
        return self._sub_problem.is_dynamic()

    def is_multi_objective(self) -> bool:
        """Returns whether the real sub-problem is multi-objective."""
        return self._sub_problem.is_multi_objective()


class CooperativeDecompositionSolver(Solver):
    """Abstract coordinator that splits a problem into groups of variables.

    Every group gets its own sub-solver, which can be any cilpy solver. A
    sub-solver evaluates its part of a solution by inserting it into the
    shared context vector, which holds the best value found for every
    variable. Subclasses decide what happens to the groups during a run by
    implementing _post_step_hook.
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
        only_accept_improvements: bool = False,
        **kwargs,
    ):
        """Initializes the cooperative solver.

        Args:
            problem: The full problem to solve.
            name: The name of the solver.
            num_groups: The number of groups to start with. Must be between
                1 and the problem dimension.
            sub_solver_class: The solver class used for every group, for
                example PSO, GA or DE.
            sub_solver_params: The parameters passed to every sub-solver.
            grouping_strategy: Decides how the variables are put into groups.
            constraint_handler: Optional constraint handler for the coordinator.
            only_accept_improvements: If True, a group's new best is only
                written into the context when it does not make the context worse.

        Raises:
            ValueError: If num_groups is not between 1 and the problem dimension.
        """
        super().__init__(problem, name, constraint_handler)

        self._only_accept_improvements = only_accept_improvements

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

        self._counting_problem = CountingProblem(problem)

        actual_number_of_groups = len(self._groups)

        self._sub_problems: List[SubSpaceProblem] = [
            SubSpaceProblem(self._counting_problem, self._context, k, self._groups[k])
            for k in range(actual_number_of_groups)
        ]

        self._sub_solvers: List[Solver] = self._instantiate_sub_solvers(
            sub_solver_class, sub_solver_params, actual_number_of_groups
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
        evaluate_starting_population: bool = True,
    ) -> List[Solver]:
        """Creates one sub-solver for each group.

        Args:
            sub_solver_class: The solver class to create.
            sub_solver_params: The parameters passed to every sub-solver.
            num_groups: The number of sub-solvers to create.
            evaluate_starting_population: If False, the sub-solvers are built
                on a PlaceholderProblem so their starting population costs no
                evaluations. Used during a regroup.

        Returns:
            One sub-solver per group, each linked to its sub-problem.
        """
        solvers = []
        base_name = sub_solver_params.get("name", self.name)
        for k in range(num_groups):
            params = dict(sub_solver_params)
            params["name"] = f"{base_name}_sub{k}"

            if evaluate_starting_population:
                solver = sub_solver_class(problem=self._sub_problems[k], **params)
            else:
                solver = sub_solver_class(problem=PlaceholderProblem(self._sub_problems[k]), **params)
                solver.problem = self._sub_problems[k]

            solvers.append(solver)
        return solvers

    def _sync_context(self) -> None:
        """Writes each sub-solver's best partial solution into the context."""
        for k, sub_solver in enumerate(self._sub_solvers):
            best_sub_solution, _ = sub_solver.get_result()[0]
            self._context.update_group(k, best_sub_solution)

    def _get_global_best(self) -> Tuple[List[float], Evaluation]:
        """Evaluates the context vector on the original problem.

        Returns:
            A tuple of the context vector and its evaluation.
        """
        full_vector = self._context.get_full_vector()
        evaluation = self.problem.evaluate(full_vector)
        return (full_vector, evaluation)

    def step(self) -> None:
        """Performs one iteration.

        Every sub-solver takes one step in turn. After each sub-solver's step
        its best partial solution gets written into the context, so the next
        sub-solver already works with it. The subclass hook runs at the end.
        """
        self._iteration += 1

        assembled_evaluation = self._counting_problem.evaluate(self._context.get_full_vector())

        for k, sub_solver in enumerate(self._sub_solvers):
            # The context may have changed since this sub-solver's last step
            sub_solver.refresh_evaluations()

            # The sub-solver's best is never worse than its part of the context
            full_vector = self._context.get_full_vector()
            context_part = [full_vector[d] for d in self._groups[k]]
            sub_solver.accept_best(context_part, assembled_evaluation)

            sub_solver.step()
            best_sub_solution, best_sub_evaluation = sub_solver.get_result()[0]

            if self._accepts_context_update(best_sub_evaluation.fitness, assembled_evaluation.fitness):
                self._context.update_group(k, best_sub_solution)
                assembled_evaluation = best_sub_evaluation

        self._best_fitness_history.append(assembled_evaluation.fitness)

        self._post_step_hook()

    def _accepts_context_update(self, candidate_fitness: float,
        current_context_fitness: float) -> bool:
        """This decides whether a group's new best is written into the context.

        Args:
            candidate_fitness: The assembled fitness if the groups new best is
                written into the context.
            current_context_fitness: Assembled fitness of the current context.

        Returns:
            True if the update should be written into the context.
        """

        if self._only_accept_improvements:
            return candidate_fitness <= current_context_fitness

        return True

    @abstractmethod
    def _post_step_hook(self) -> None:
        """Runs at the end of each step.

        Subclasses use it to decide when and how the groups change, for
        example by regrouping, splitting or merging them.
        """

    def _build_state_dict(self) -> dict:
        """
        Builds the state dictionary used by a regroup trigger.

        Returns:
            A dictionary with the current iteration count, the best
            assembled fitness reached so far and the number of
            evaluations used so far.
        """
        if self._best_fitness_history:
            best_fitness = self._best_fitness_history[-1]

        else:
            best_fitness = float("inf")

        return {
            "iteration": self._iteration,
            "best_fitness": best_fitness,
            "evaluations": self._counting_problem.evaluations,
        }

    def _require_uniform_population_size(self) -> None:
        """
        Checks that every sub-solver holds the same number of individuals.

        The state is copied column by column during a regroup, so every
        sub-solver needs the same number of rows.

        Raises:
            ValueError: If the sub-solvers hold different numbers of individuals.
        """
        population_sizes = set()

        for sub_solver in self._sub_solvers:
            population_sizes.add(len(sub_solver.get_population()))

        if len(population_sizes) > 1:
            raise ValueError(
                "regrouping needs every sub-solver to hold the same number of "
                f"individuals, found sizes {sorted(population_sizes)}"
            )

    def _collect_state_columns(self, attribute_name: str) -> Optional[dict]:
        """
        Reads one sub-solver state matrix into per-dimension columns.

        Args:
            attribute_name: Name of a sub-solver attribute, for example
                population or velocities.

        Returns:
            A dictionary mapping each global dimension index to the list of
            values the individuals hold for that dimension. None if a
            sub-solver does not have the attribute.
        """
        for sub_solver in self._sub_solvers:
            if not hasattr(sub_solver, attribute_name):
                return None

        columns_by_dimension = {}

        for k, sub_solver in enumerate(self._sub_solvers):
            state_matrix = getattr(sub_solver, attribute_name)

            for local_idx, global_idx in enumerate(self._groups[k]):
                
                column = [row[local_idx] for row in state_matrix]
                columns_by_dimension[global_idx] = column

        return columns_by_dimension

    def _apply_state_columns(self, attribute_name: str, columns_by_dimension: dict) -> None:
        """
        Writes per-dimension columns back into the freshly built sub-solvers.

        Args:
            attribute_name: Name of the sub solver attribute to write into.
            columns_by_dimension: Columns created by _collect_state_columns.
        """
        for k, sub_solver in enumerate(self._sub_solvers):
            state_matrix = getattr(sub_solver, attribute_name)

            for local_idx, global_idx in enumerate(self._groups[k]):
                column = columns_by_dimension[global_idx]

                for individual_idx, value in enumerate(column):
                    state_matrix[individual_idx][local_idx] = value

    def _rebuild_sub_problems(self, new_groups: List[List[int]]) -> None:
        """
        Makes the list of sub-problems match the new grouping.

        Existing sub-problems are reused with their new indices, extra ones
        are removed and missing ones are created.

        Args:
            new_groups: The new assignment of dimensions into groups.
        """
        del self._sub_problems[len(new_groups):]

        for k, group in enumerate(new_groups):
            if k < len(self._sub_problems):
                self._sub_problems[k].update_dim_indices(group)

            else:
                self._sub_problems.append(
                    SubSpaceProblem(self._counting_problem, self._context, k, group))

    def _perform_regroup(self, new_groups: List[List[int]]) -> None:
        """
        Replaces the current grouping with a new grouping of all the dimensions.

        The positions, velocities and personal bests are saved per dimension,
        new sub-solvers are built without using evaluations, and the saved
        state is written back in the new arrangement. The context vector
        keeps its values, so no search progress is lost.

        Args:
            new_groups: The new assignment of all dimensions into groups.

        Raises:
            ValueError: If the sub-solvers hold different numbers of individuals.
        """

        self._require_uniform_population_size()

        preserved_state = {}

        for attribute_name in PER_DIMENSION_STATE_ATTRIBUTES:
            
            columns_by_dimension = self._collect_state_columns(attribute_name)

            if columns_by_dimension is not None:
                preserved_state[attribute_name] = columns_by_dimension


        self._groups = [list(group) for group in new_groups]
        self._context.reassign_groups(new_groups)

        self._rebuild_sub_problems(new_groups)

        self._sub_solvers = self._instantiate_sub_solvers(
            self._sub_solver_class, self._sub_solver_params, len(new_groups),
            evaluate_starting_population=False)

        for attribute_name, columns_by_dimension in preserved_state.items():
            self._apply_state_columns(attribute_name, columns_by_dimension)



        self._last_regroup_iteration = self._iteration

    def _give_new_velocities(self) -> None:
        """Gives every particle a new random velocity.

        Used by DCPSO after a split. Each velocity is drawn from
        [-range, range] of its dimension. Solvers without velocities are skipped.
        """

        for sub_solver in self._sub_solvers:


            if not hasattr(sub_solver, "velocities"):
                continue

            lower_bounds, upper_bounds = sub_solver.problem.bounds

            for velocity in sub_solver.velocities:
                
                for d in range(len(velocity)):
                    
                    span_size = upper_bounds[d] - lower_bounds[d]
                    velocity[d] = random.uniform(-span_size, span_size)

    def _give_new_positions(self) -> None:
        """Gives new random positions to GA and DE individuals.

        Used by DCPSO after a split. The first individual is kept and the
        rest are placed randomly inside the bounds. Solvers with velocities
        are skipped.
        """

        for sub_solver in self._sub_solvers:

            if hasattr(sub_solver, "velocities"):
                continue

            lower_bounds, upper_bounds = sub_solver.problem.bounds

            for individual in sub_solver.population[1:]:

                for d in range(len(individual)):
                    individual[d] = random.uniform(lower_bounds[d], upper_bounds[d])

    def get_population(self) -> List[List[float]]:
        """Returns every sub-solver's individuals as full-dimension vectors.

        Each individual is inserted into a copy of the context vector.

        Returns:
            One full-dimension vector per individual of every sub-solver.
        """
        expanded = []

        for k, sub_solver in enumerate(self._sub_solvers):
            for sub_individual in sub_solver.get_population():
                full = self._context.get_full_vector()

                for local_idx, global_idx in enumerate(self._groups[k]):
                    full[global_idx] = sub_individual[local_idx]

                expanded.append(full)
        return expanded

    def get_population_evaluations(self) -> List[Evaluation]:
        """Evaluates the full-dimension population on the original problem.

        This costs one evaluation per individual. The runner calls it inside
        not_counting, so it does not use up the evaluation budget there.

        Returns:
            The evaluation of every vector from get_population.
        """

        evaluations = []
        for individual in self.get_population():
            evaluation = self.problem.evaluate(individual)
            evaluations.append(evaluation)
        return evaluations

    def get_result(self) -> List[Tuple[List[float], Evaluation]]:
        """Returns the context vector and its evaluation as the best solution.

        Returns:
            A list with one tuple of the context vector and its evaluation.
        """
        return [self._get_global_best()]