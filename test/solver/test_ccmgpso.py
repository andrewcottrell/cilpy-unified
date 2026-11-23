# test/solver/test_ccmgpso.py
"""Tests for the CoevolutionaryLagrangianSolver with MGPSO as the objective
solver (CCMGPSO), under both archive strategies, and for its
single-objective path."""

import numpy as np
import pytest

from cilpy.problem.multi_objective import CONSTR
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.ga import GA
from cilpy.solver.pso import PSO
from cilpy.solver.ccls import (
    CoevolutionaryLagrangianSolver,
    _total_violation,
    _LagrangianMinProblem,
)
from cilpy.problem.constrained import G01


def _make_ccmgpso(problem, strategy, swarm_size=20):
    return CoevolutionaryLagrangianSolver(
        name="CCMGPSO",
        problem=problem,
        objective_solver_class=MGPSO,
        multiplier_solver_class=PSO,
        objective_solver_params={"swarm_size": swarm_size},
        multiplier_solver_params={
            "swarm_size": 30, "w": 0.40, "c1": 1.20, "c2": 1.20,
        },
        penalty_rho=1.0,
        multiplier_max=1000.0,
        archive_strategy=strategy,
    )


class TestMinProxyMO:
    def test_mo_proxy_returns_vector_fitness(self):
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        ev = proxy.evaluate([0.5, 1.0])
        assert isinstance(ev.fitness, list)
        assert len(ev.fitness) == 2

    def test_penalty_added_equally_to_all_objectives(self):
        problem = CONSTR()
        proxy = _LagrangianMinProblem(problem, penalty_rho=1.0)
        proxy.set_fixed_multipliers([2.0, 2.0], [])
        x = [0.15, 0.0]  # infeasible: g1 = 6 - 0 - 1.35 > 0
        raw = problem.evaluate(x)
        pen = proxy.evaluate(x)
        deltas = [p - r for p, r in zip(pen.fitness, raw.fitness)]
        assert deltas[0] == pytest.approx(deltas[1])
        assert deltas[0] > 0  # infeasible -> positive penalty

    def test_feasible_solution_penalty_matches_lagrangian_slack(self):
        problem = CONSTR()
        proxy = _LagrangianMinProblem(problem, penalty_rho=1.0)
        proxy.set_fixed_multipliers([0.0, 0.0], [])
        x = [1.0, 0.0]  # feasible, zero multipliers -> no penalty
        raw = problem.evaluate(x)
        pen = proxy.evaluate(x)
        assert pen.fitness == pytest.approx(raw.fitness)

    def test_mo_proxy_dynamism_delegated(self):
        # Multiplier changes are not environment changes; the solver re-scores
        # the objective solver directly instead (MGPSO.rescore).
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        assert proxy.is_dynamic() == CONSTR().is_dynamic()

    def test_so_proxy_dynamism_unchanged(self):
        proxy = _LagrangianMinProblem(G01(), penalty_rho=1.0)
        assert proxy.is_dynamic() == G01().is_dynamic()

    def test_constraints_passed_through(self):
        proxy = _LagrangianMinProblem(CONSTR(), penalty_rho=1.0)
        ev = proxy.evaluate([0.15, 0.0])
        assert ev.constraints_inequality is not None
        assert len(ev.constraints_inequality) == 2


class TestCCMGPSOIntegration:
    @pytest.mark.parametrize("strategy", ["filter", "strict"])
    def test_runs_and_returns_front(self, strategy):
        np.random.seed(3)
        solver = _make_ccmgpso(CONSTR(), strategy)
        for _ in range(30):
            solver.step()
        result = solver.get_result()
        assert len(result) >= 1
        # Results are re-evaluated on the original problem: no penalty.
        for solution, ev in result:
            fresh = CONSTR().evaluate(solution)
            assert np.allclose(fresh.fitness, ev.fitness)

    def test_strict_archive_contains_only_feasible(self):
        np.random.seed(3)
        solver = _make_ccmgpso(CONSTR(), "strict")
        for _ in range(30):
            solver.step()
        # The MGPSO archive holds feasible solutions only.
        for _, ev in solver.objective_solver.get_result():
            assert _total_violation(ev) == 0.0

    def test_filter_mode_final_front_feasible(self):
        np.random.seed(3)
        solver = _make_ccmgpso(CONSTR(), "filter")
        for _ in range(60):
            solver.step()
        result = solver.get_result()
        feasibility = [
            _total_violation(ev) == 0.0 for _, ev in result
        ]
        # CONSTR has a feasible front after 60 steps.
        assert all(feasibility)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            _make_ccmgpso(CONSTR(), "banana")

    def test_single_objective_path_unaffected(self):
        np.random.seed(3)
        solver = CoevolutionaryLagrangianSolver(
            name="CCLS",
            problem=G01(),
            objective_solver_class=PSO,
            multiplier_solver_class=PSO,
            objective_solver_params={
                "swarm_size": 20, "w": 0.72, "c1": 1.49, "c2": 1.49,
            },
            multiplier_solver_params={
                "swarm_size": 20, "w": 0.40, "c1": 1.20, "c2": 1.20,
            },
            penalty_rho=0.5,
            multiplier_max=10000.0,
        )
        for _ in range(10):
            solver.step()
        result = solver.get_result()
        assert len(result) == 1
        assert isinstance(result[0][1].fitness, float)


class TestMultiplierEvolution:
    def test_multipliers_change_over_run(self):
        """The per-objective multipliers change over a run."""
        np.random.seed(42)
        solver = _make_ccmgpso(CONSTR(), "filter", swarm_size=20)
        seen = set()
        for _ in range(50):
            solver.step()
            seen.add(str(solver.min_problem.objective_multipliers))
        assert len(seen) > 1


class TestChangeResponseHook:
    """The framework must work with any solver, not only PSO."""

    @pytest.mark.parametrize("objective_class, objective_params", [
        (GA, {"population_size": 10, "crossover_rate": 0.8,
              "mutation_rate": 0.1}),
        (PSO, {"swarm_size": 10, "w": 0.7, "c1": 1.4, "c2": 1.4}),
    ])
    def test_runs_with_ga_multiplier_solver(self, objective_class,
                                            objective_params):
        solver = CoevolutionaryLagrangianSolver(
            name="CCLS",
            problem=G01(),
            objective_solver_class=objective_class,
            multiplier_solver_class=GA,
            objective_solver_params=objective_params,
            multiplier_solver_params={
                "population_size": 10, "crossover_rate": 0.8,
                "mutation_rate": 0.1,
            },
        )
        for _ in range(5):
            solver.step()
        assert len(solver.get_result()) == 1

    def test_pso_rescoring_follows_the_landscape(self):
        """PSO re-scores its bests against the problem as it now stands."""
        solver = CoevolutionaryLagrangianSolver(
            name="CCLS",
            problem=G01(),
            objective_solver_class=PSO,
            multiplier_solver_class=PSO,
            objective_solver_params={
                "swarm_size": 10, "w": 0.7, "c1": 1.4, "c2": 1.4,
            },
            multiplier_solver_params={
                "swarm_size": 10, "w": 0.4, "c1": 1.2, "c2": 1.2,
            },
        )
        solver.step()
        ms = solver.multiplier_solver
        ms.respond_to_change()
        for position, evaluation in zip(ms.pbest_positions,
                                        ms.pbest_evaluations):
            fresh = solver.max_problem.evaluate(position)
            assert evaluation.fitness == pytest.approx(fresh.fitness)
        assert not any(
            ms.comparator.is_better(evaluation, ms.gbest_evaluation)
            for evaluation in ms.pbest_evaluations
        )
