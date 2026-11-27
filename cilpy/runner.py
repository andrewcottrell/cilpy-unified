# cilpy/runner.py
"""The experiment runner: Orchestrates computational intelligence experiments."""

import time
import csv
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Sequence

from cilpy.problem import Problem, Evaluation
from cilpy.problem.counting import CountingProblem
from cilpy.solver import Solver
from cilpy.compare.metrics import (
    inverted_generational_distance,
    generational_distance,
    hypervolume,
    hypervolume_ratio,
    p_red,
    spacing,
    spread,
)


class ExperimentRunner:
    """Runs every solver configuration on every problem and writes the results.

    For each (problem, solver) pair, two CSV files are written to
    ``output_dir`` (default ``out``).

    ``<problem>_<solver>.out.csv`` has one row per iteration of every run.
        Single-objective: best fitness, population feasibility (%),
        population diversity (mean distance to the swarm centroid) and
        relative error (f_max - f) / (f_max - f_min), whether the best
        solution is feasible, and the fitness bounds.
        Multi-objective: archive size, IGD, GD, hypervolume, spacing, spread,
        population feasibility and diversity, the hypervolume ratio of the
        feasible archive to the true front (the score used by P_RED), and the
        archive's objective vectors. IGD, GD, spread and the hypervolume ratio
        need ``true_pareto_front()`` and are left empty without it.

    ``<problem>_<solver>.summary.out.csv`` has one row per run: the final
        values of the columns above, the front feasibility (%), the feasible
        front, P_RED, the run time, and the number of problem evaluations made
        by the solver (evaluations made to measure the result are not counted).

    Both files end with an ``evaluations`` column: the solver's evaluations
    so far (per iteration) or in total (per run).

    A run stops after ``max_iterations`` iterations or ``max_evaluations``
    solver evaluations, whichever comes first. At least one must be given.
    """

    def __init__(
        self,
        problems: Sequence[Problem],
        solver_configurations: List[Dict[str, Any]],
        num_runs: int,
        max_iterations: Optional[int] = None,
        output_dir: str = "out",
        max_evaluations: Optional[int] = None,
    ):
        """
        Initializes the ExperimentRunner.

        Args:
            problems: A sequence of problem instances to be solved.
                Each object must implement the `Problem` interface.
            solver_configurations: A list of solver configurations.
                Each configuration is a dictionary specifying the solver class
                and its parameters. The `problem` parameter is injected
                automatically by the runner and should not be included.
            num_runs: The number of independent runs for each
                solver-problem pair.
            max_iterations: The iteration budget (`solver.step()` calls) of a
                run.
            output_dir: Directory for the result CSVs. Defaults to "out",
                relative to the current working directory.
            max_evaluations: The fitness evaluation budget of a run. It is
                checked between iterations, so a run can use slightly more.
                If both budgets are given, a run stops at whichever is
                reached first.

        Raises:
            ValueError: If no budget is given or a budget is less than 1.

        Solver Configuration Format:
            .. code-block:: python

                [
                    {
                        "class": YourSolverClass,
                        "params": {
                            "name": "UniqueSolverName",
                            "param1": value1,
                            # ... other solver hyperparameters
                        }
                    },
                    # ... more configurations
                ]
        """
        if max_iterations is None and max_evaluations is None:
            raise ValueError("either max_iterations or max_evaluations must be given")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, but got {max_iterations}")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(f"max_evaluations must be at least 1, but got {max_evaluations}")

        self.problems = problems
        self.solver_configurations = solver_configurations
        self.num_runs = num_runs
        self.max_iterations = max_iterations
        self.max_evaluations = max_evaluations
        self.output_dir = output_dir

    # Public interface

    def run_experiments(self):
        """Executes the full suite of defined experiments."""
        total_start_time = time.time()
        print("======== Starting All Experiments ========")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        for problem in self.problems:
            print(f"\n--- Processing Problem: {problem.name} ---")
            for config in self.solver_configurations:
                solver_class = config["class"]
                solver_params = config["params"].copy()
                constraint_handler_config = config.get("constraint_handler")

                solver_params["problem"] = problem
                solver_name = solver_params.get("name", solver_class.__name__)
                output_file_path = os.path.join(
                    self.output_dir, f"{problem.name}_{solver_name}.out.csv"
                )

                print(f"\n  -> Starting Experiment: {solver_name} on {problem.name}")
                print(
                    f"     Configuration: {self.num_runs} runs, "
                    f"{self._budget_description()}."
                )
                print(f"     Results will be saved to: {output_file_path}")

                self._run_single_experiment(
                    solver_class,
                    solver_params,
                    output_file_path,
                    constraint_handler_config,
                )

        total_end_time = time.time()
        print("\n======== All Experiments Finished ========")
        print(f"Total execution time: {total_end_time - total_start_time:.2f}s")

    # Shared helpers

    def _budget_description(self) -> str:
        """Returns the run budget as text for the console output."""
        parts = []
        if self.max_iterations is not None:
            parts.append(f"{self.max_iterations} iterations/run")
        if self.max_evaluations is not None:
            parts.append(f"{self.max_evaluations} evaluations/run")
        return " or ".join(parts)

    def _budget_is_used(self, iteration: int, problem: CountingProblem) -> bool:
        """Returns True once the iteration or evaluation budget is reached."""
        if self.max_iterations is not None and iteration >= self.max_iterations:
            return True
        if self.max_evaluations is not None and problem.evaluations >= self.max_evaluations:
            return True
        return False

    def _is_solution_feasible(self, evaluation: Evaluation, tolerance: float = 1e-6) -> bool:
        if evaluation is None:
            return False
        if evaluation.constraints_inequality:
            if any(v > 0 for v in evaluation.constraints_inequality):
                return False
        if evaluation.constraints_equality:
            if any(abs(v) > tolerance for v in evaluation.constraints_equality):
                return False
        return True

    def _measure_feasibility(self, solver: Solver) -> str:
        try:
            evals = solver.get_population_evaluations()
            if not evals:
                return ""
            n_feasible = sum(1 for e in evals if self._is_solution_feasible(e))
            return f"{100.0 * n_feasible / len(evals):.4f}"
        except NotImplementedError:
            return ""

    def _measure_diversity(self, solver: Solver) -> str:
        try:
            population = solver.get_population()
            if not population:
                return ""
            pop_array = np.array(population, dtype=float)
            centroid = np.mean(pop_array, axis=0)
            dists = np.sqrt(np.sum((pop_array - centroid) ** 2, axis=1))
            return f"{float(np.mean(dists)):.6f}"
        except NotImplementedError:
            return ""

    # Single-objective columns

    _SO_MAIN_HEADER = [
        "run_id", "iteration", "best_fitness", "population_feasibility_pct",
        "population_diversity", "relative_error", "is_feasible",
        "optimum_value", "worst_value", "evaluations",
    ]

    _SO_SUMMARY_HEADER = [
        "problem_name", "solver_name", "run_id", "final_best_fitness",
        "final_feasibility_pct", "p_red", "run_time_s", "evaluations",
    ]

    def _so_iteration_row(
        self,
        run_id: int,
        iteration: int,
        solver: Solver,
        bounds_known: bool,
        f_min: float,
        f_max: float,
        fitness_range: float,
        relative_error_history: List[float],
    ) -> list:
        result = solver.get_result()
        best_fitness = result[0][1].fitness if result else ""

        if bounds_known and isinstance(best_fitness, (int, float)):
            rel_err = (f_max - best_fitness) / fitness_range
            relative_error_history.append(rel_err)
            rel_err_str = f"{rel_err:.6f}"
        else:
            rel_err_str = ""

        return [
            run_id,
            iteration,
            f"{best_fitness:.6f}" if isinstance(best_fitness, (int, float)) else best_fitness,
            self._measure_feasibility(solver),
            self._measure_diversity(solver),
            rel_err_str,
            int(self._is_solution_feasible(result[0][1])) if result else "",
            f_min if bounds_known else "",
            f_max if bounds_known else "",
        ]

    def _so_summary_row(
        self,
        problem_name: str,
        solver_name: str,
        run_id: int,
        solver: Solver,
        relative_error_history: List[float],
        run_time: float,
        evaluations: int,
    ) -> list:
        result = solver.get_result()
        final_fitness = result[0][1].fitness if result else ""
        final_feasibility = self._measure_feasibility(solver)

        if relative_error_history:
            p_red_str = f"{p_red(relative_error_history):.6f}"
        else:
            p_red_str = ""

        return [
            problem_name,
            solver_name,
            run_id,
            f"{final_fitness:.6f}" if isinstance(final_fitness, (int, float)) else final_fitness,
            final_feasibility,
            p_red_str,
            f"{run_time:.2f}",
            evaluations,
        ]

    # Multi-objective columns

    _MO_MAIN_HEADER = [
        "run_id", "iteration", "archive_size", "igd", "gd", "hypervolume",
        "spacing", "spread", "population_feasibility_pct",
        "population_diversity", "relative_error", "front", "evaluations",
    ]

    _MO_SUMMARY_HEADER = [
        "problem_name", "solver_name", "run_id", "final_archive_size",
        "final_igd", "final_gd", "final_hypervolume", "final_spacing",
        "final_spread", "final_feasibility_pct", "final_front_feasibility_pct",
        "final_feasible_front", "p_red", "run_time_s", "evaluations",
    ]

    def _front_metrics(self, front, hv_ref, ref_front) -> Tuple[str, ...]:
        """Returns (igd, gd, hypervolume, spacing, spread) of a front as strings.

        A metric that cannot be computed is an empty string.
        """
        def fmt(metric, *args):
            try:
                return f"{metric(front, *args):.6f}"
            except (ValueError, NotImplementedError):
                return ""

        if not front:
            return ("",) * 5
        hv = fmt(hypervolume, hv_ref) if hv_ref else ""
        sp = fmt(spacing)
        if ref_front is None:
            return "", "", hv, sp, ""
        return (
            fmt(inverted_generational_distance, ref_front),
            fmt(generational_distance, ref_front),
            hv,
            sp,
            fmt(spread, ref_front),
        )

    def _mo_iteration_row(
        self,
        run_id: int,
        iteration: int,
        solver: Solver,
        hv_ref: Optional[Tuple[float, float]],
        ref_front: Optional[np.ndarray],
        relative_error_history: List[float],
    ) -> Tuple[list, Optional[Tuple[float, float]]]:
        """Returns (row, updated_hv_ref)."""
        result = solver.get_result()
        front_fitnesses = [ev.fitness for _, ev in result]

        archive_size = len(front_fitnesses)

        # HV reference point: set from the first non-empty archive of the run.
        if hv_ref is None and front_fitnesses:
            f1_max = max(f[0] for f in front_fitnesses)
            f2_max = max(f[1] for f in front_fitnesses)
            hv_ref = (f1_max * 1.1 + 0.1, f2_max * 1.1 + 0.1)

        igd_str, gd_str, hv_str, sp_str, spread_str = self._front_metrics(
            front_fitnesses, hv_ref, ref_front
        )

        # Relative error: HV ratio of the feasible archive to the true front.
        red_str = ""
        if ref_front is not None:
            feasible_fitnesses = [
                ev.fitness for _, ev in result
                if self._is_solution_feasible(ev)
            ]
            try:
                b = hypervolume_ratio(feasible_fitnesses, ref_front)
                relative_error_history.append(b)
                red_str = f"{b:.6f}"
            except (ValueError, NotImplementedError):
                pass

        row = [
            run_id,
            iteration,
            archive_size,
            igd_str,
            gd_str,
            hv_str,
            sp_str,
            spread_str,
            self._measure_feasibility(solver),
            self._measure_diversity(solver),
            red_str,
            [[float(v) for v in f] for f in front_fitnesses],
        ]
        return row, hv_ref

    def _mo_summary_row(
        self,
        problem_name: str,
        solver_name: str,
        run_id: int,
        solver: Solver,
        hv_ref: Optional[Tuple[float, float]],
        ref_front: Optional[np.ndarray],
        relative_error_history: List[float],
        run_time: float,
        evaluations: int,
    ) -> list:
        result = solver.get_result()
        front_fitnesses = [ev.fitness for _, ev in result]

        igd_str, gd_str, hv_str, sp_str, spread_str = self._front_metrics(
            front_fitnesses, hv_ref, ref_front
        )

        # Feasibility of the returned front.
        front_evals = [ev for _, ev in result]
        n_front_feasible = sum(
            1 for ev in front_evals if self._is_solution_feasible(ev)
        )
        front_feas_str = (
            f"{100.0 * n_front_feasible / len(front_evals):.4f}"
            if front_evals else ""
        )
        feasible_front = [
            [float(v) for v in ev.fitness]
            for ev in front_evals
            if self._is_solution_feasible(ev)
        ]

        p_red_str = (
            f"{p_red(relative_error_history):.6f}"
            if relative_error_history else ""
        )

        return [
            problem_name,
            solver_name,
            run_id,
            len(front_fitnesses),
            igd_str,
            gd_str,
            hv_str,
            sp_str,
            spread_str,
            self._measure_feasibility(solver),
            front_feas_str,
            feasible_front,
            p_red_str,
            f"{run_time:.2f}",
            evaluations,
        ]

    # run loop

    def _run_single_run(
        self,
        run_id: int,
        constraint_handler_config: Optional[Dict],
        solver_params: Dict,
        solver_class: Type[Solver],
        main_writer,
        summary_file_path: str,
        is_multi_objective: bool,
        bounds_known: bool = False,
        f_min: float = 0.0,
        f_max: float = 0.0,
        fitness_range: float = 0.0,
        ref_front: Optional[np.ndarray] = None,
    ):
        run_start_time = time.time()
        print(f"     --- Starting Run {run_id}/{self.num_runs} ---")

        # Each run of a dynamic problem starts at t = 0.
        problem_for_run = solver_params["problem"]
        if hasattr(problem_for_run, "reset_time"):
            problem_for_run.reset_time()

        current_params = solver_params.copy()

        # Pass constraint_handler only when one is configured.
        if constraint_handler_config:
            handler_class = constraint_handler_config["class"]
            handler_params = constraint_handler_config.get("params", {})
            current_params["constraint_handler"] = handler_class(**handler_params)

        # Count the solver's problem evaluations with a fresh counter per run.
        # Counting is switched off while the runner measures the result.
        counted_problem = CountingProblem(problem_for_run)
        current_params["problem"] = counted_problem
        solver = solver_class(**current_params)
        with counted_problem.not_counting():
            self._iterate(
                solver, counted_problem, run_id, run_start_time, main_writer,
                summary_file_path, is_multi_objective, bounds_known, f_min,
                f_max, fitness_range, ref_front,
            )

    def _iterate(
        self,
        solver: Solver,
        counted_problem: CountingProblem,
        run_id: int,
        run_start_time: float,
        main_writer,
        summary_file_path: str,
        is_multi_objective: bool,
        bounds_known: bool,
        f_min: float,
        f_max: float,
        fitness_range: float,
        ref_front: Optional[np.ndarray],
    ):

        problem_name = solver.problem.name
        solver_name = solver.name

        relative_error_history: List[float] = []
        hv_ref: Optional[Tuple[float, float]] = None

        iteration = 0
        while not self._budget_is_used(iteration, counted_problem):
            iteration += 1
            evaluations_before_step = counted_problem.evaluations
            solver.problem.begin_iteration()
            with counted_problem.counting():
                solver.step()

            # Without this check a solver that never evaluates would loop forever.
            if (counted_problem.evaluations == evaluations_before_step
                    and self.max_iterations is None):
                raise RuntimeError(
                    f"{solver.name} made no fitness evaluations in iteration "
                    f"{iteration}, so the evaluation budget can never be "
                    "reached. Set max_iterations as well."
                )

            if is_multi_objective:
                # Dynamic problems: refresh the true front every iteration.
                if ref_front is not None and any(solver.problem.is_dynamic()):
                    try:
                        ref_front = solver.problem.true_pareto_front()
                    except (NotImplementedError, AttributeError):
                        pass
                row, hv_ref = self._mo_iteration_row(
                    run_id, iteration, solver, hv_ref, ref_front,
                    relative_error_history,
                )
            else:
                row = self._so_iteration_row(
                    run_id, iteration, solver,
                    bounds_known, f_min, f_max, fitness_range,
                    relative_error_history,
                )

            row.append(counted_problem.evaluations)
            main_writer.writerow(row)

        run_time = time.time() - run_start_time

        # Console summary
        if is_multi_objective:
            result = solver.get_result()
            front_fitnesses = [ev.fitness for _, ev in result]
            igd_console = ""
            if ref_front is not None and front_fitnesses:
                try:
                    igd_console = f"  IGD={inverted_generational_distance(front_fitnesses, ref_front):.4f}"
                except (ValueError, NotImplementedError):
                    pass
            p_red_console = ""
            if relative_error_history:
                p_red_console = f"  P_RED={p_red(relative_error_history):.6f}"
            print(
                f"     Run {run_id} finished in {run_time:.2f}s. "
                f"Archive size: {len(result)}{igd_console}{p_red_console}"
            )
        else:
            result = solver.get_result()
            final_fitness = result[0][1].fitness if result else "N/A"
            print(
                f"     Run {run_id} finished in {run_time:.2f}s. "
                f"Best fitness: {final_fitness}"
            )
            if relative_error_history:
                print(f"     P_RED for Run {run_id}: {p_red(relative_error_history):.6f}")
            else:
                print(f"     P_RED for Run {run_id}: N/A (fitness bounds unknown)")

        # Append to the summary file
        with open(summary_file_path, "a", newline="") as f_sum:
            w = csv.writer(f_sum)
            if is_multi_objective:
                w.writerow(self._mo_summary_row(
                    problem_name, solver_name, run_id, solver,
                    hv_ref, ref_front, relative_error_history, run_time,
                    counted_problem.evaluations,
                ))
            else:
                w.writerow(self._so_summary_row(
                    problem_name, solver_name, run_id, solver,
                    relative_error_history, run_time, counted_problem.evaluations,
                ))

    def _run_single_experiment(
        self,
        solver_class: Type[Solver],
        solver_params: Dict,
        output_file: str,
        constraint_handler_config: Optional[Dict] = None,
    ):
        problem = solver_params["problem"]
        is_mo = problem.is_multi_objective()

        # Single-objective: fitness bounds for the relative error
        bounds_known = False
        f_min = 0.0
        f_max = 0.0
        fitness_range = 0.0
        if not is_mo:
            try:
                f_min, f_max = problem.get_fitness_bounds()
                fitness_range = f_max - f_min
                if fitness_range > 0:
                    bounds_known = True
                else:
                    print("     Warning: Fitness range is zero; relative error skipped.")
            except NotImplementedError:
                print("     Info: get_fitness_bounds() not implemented; relative error skipped.")

        # Multi-objective: true front, computed once before the runs
        ref_front: Optional[np.ndarray] = None
        if is_mo:
            try:
                print("     Generating reference Pareto front...", end=" ", flush=True)
                ref_front = problem.true_pareto_front()
                print(f"done ({len(ref_front)} points). IGD/GD/Spread will be reported.")
            except (NotImplementedError, AttributeError):
                print("     Info: true_pareto_front() not available; IGD/GD/Spread will be empty.")

        summary_file_path = output_file.replace(".out.csv", ".summary.out.csv")
        main_header = self._MO_MAIN_HEADER if is_mo else self._SO_MAIN_HEADER
        summary_header = self._MO_SUMMARY_HEADER if is_mo else self._SO_SUMMARY_HEADER

        with open(summary_file_path, "w", newline="") as f_sum:
            csv.writer(f_sum).writerow(summary_header)

        experiment_start_time = time.time()

        with open(output_file, "w", newline="") as f_main:
            writer = csv.writer(f_main)
            writer.writerow(main_header)

            for run_id in range(1, self.num_runs + 1):
                self._run_single_run(
                    run_id,
                    constraint_handler_config,
                    solver_params,
                    solver_class,
                    writer,
                    summary_file_path,
                    is_multi_objective=is_mo,
                    bounds_known=bounds_known,
                    f_min=f_min,
                    f_max=f_max,
                    fitness_range=fitness_range,
                    ref_front=ref_front,
                )

        experiment_time = time.time() - experiment_start_time
        solver_name = solver_params.get("name", solver_class.__name__)
        print(
            f"  -> Experiment for {solver_name} on {problem.name} "
            f"finished in {experiment_time:.2f}s."
        )
        print(f"     Summary saved to: {summary_file_path}")


if __name__ == "__main__":
    from cilpy.problem.unconstrained import Sphere
    from cilpy.solver.pso import PSO

    runner = ExperimentRunner(
        problems=[Sphere(dimension=3)],
        solver_configurations=[{
            "class": PSO,
            "params": {
                "name": "PSO_Standard",
                "swarm_size": 30,
                "w": 0.7298,
                "c1": 1.49618,
                "c2": 1.49618,
            },
        }],
        num_runs=5,
        max_iterations=1000,
    )
    runner.run_experiments()
