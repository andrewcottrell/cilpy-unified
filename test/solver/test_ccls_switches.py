# test/solver/test_ccls_switches.py
"""Tests for the CCLS multiplier mode, dual scoring, penalty sign and
re-score switches."""
import random

import numpy as np
import pytest

from cilpy.problem import Evaluation
from cilpy.problem.dynamic_multi_objective import DTNK
from cilpy.problem.multi_objective import TNK
from cilpy.solver.ccls import CoevolutionaryLagrangianSolver, _LagrangianMinProblem
from cilpy.solver.mgpso import MGPSO
from cilpy.solver.pso import PSO


def _solver(problem, **kwargs):
    random.seed(1)
    np.random.seed(1)
    kwargs.setdefault("archive_strategy", "strict")
    return CoevolutionaryLagrangianSolver(
        name="t", problem=problem,
        objective_solver_class=MGPSO, multiplier_solver_class=PSO,
        objective_solver_params={"swarm_size": 20, "w": 0.72, "c1": 1.49,
                                 "c2": 1.49, "c3": 1.49,
                                 "refresh_mode": "clean",
                                 "sentry_mode": "archive"},
        multiplier_solver_params={"swarm_size": 10, "w": 0.4, "c1": 1.2,
                                  "c2": 1.2},
        multiplier_max=1000.0,
        multiplier_immigrants=0.3, multiplier_steps=5, **kwargs)


def _run(solver, steps):
    for _ in range(steps):
        solver.step()
    return solver


@pytest.mark.parametrize("key, value", [
    ("multiplier_mode", "other"),
    ("dual_scoring", "other"),
    ("penalty_sign", "other"),
])
def test_invalid_switch_values_raise(key, value):
    with pytest.raises(ValueError):
        _solver(TNK(), **{key: value})


def test_defaults():
    s = _solver(TNK())
    assert s.multiplier_mode == "per_objective"
    assert s.dual_scoring == "sampled"
    assert s.min_problem.penalty_sign == "clamped"
    assert s.rescore is True


def test_explicit_defaults_match_omitted_defaults():
    a = _run(_solver(TNK()), 20).get_result()
    b = _run(_solver(TNK(), multiplier_mode="per_objective",
                     dual_scoring="sampled", penalty_sign="clamped",
                     rescore=True), 20).get_result()
    assert [x for x, _ in a] == [x for x, _ in b]


class TestSharedMode:
    def test_no_per_objective_swarms(self):
        s = _solver(TNK(), multiplier_mode="shared")
        assert s.multiplier_solvers == []
        assert s.dual_problems == []

    def test_runs_and_uses_one_vector_for_all_objectives(self):
        s = _run(_solver(TNK(), multiplier_mode="shared"), 30)
        assert s.min_problem.objective_multipliers is None
        pairs = s.objective_multipliers
        assert len(pairs) == 2
        assert pairs[0] == pairs[1]
        assert pairs[0][0] == list(s.min_problem.fixed_multipliers_inequality)

    def test_detects_environment_changes(self):
        problem = DTNK(tau_t=5, n_t=10)
        s = _solver(problem, multiplier_mode="shared")
        for _ in range(30):
            problem.begin_iteration()
            s.step()
        assert s.objective_solver.n_changes_detected > 0


class TestAnchorScoring:
    def test_reference_set_is_one_solution(self):
        s = _run(_solver(TNK(), dual_scoring="anchor"), 10)
        for dual in s.dual_problems:
            assert len(dual._f) == 1

    def test_sampled_reference_set_is_larger(self):
        s = _run(_solver(TNK()), 10)
        for dual in s.dual_problems:
            assert len(dual._f) > 1

    def test_anchor_is_most_violating_archive_member(self):
        s = _run(_solver(TNK(), archive_strategy="filter"), 10)
        results = s.objective_solver.get_result()
        anchor = s._select_multiplier_anchor()
        assert any(list(x) == list(anchor) for x, _ in results)


class TestPenaltySign:
    def _min_problem(self, sign):
        problem = _LagrangianMinProblem(TNK(), penalty_rho=0.0)
        problem.penalty_sign = sign
        return problem

    def test_clamped_ignores_satisfied_constraints(self):
        p = self._min_problem("clamped")
        assert p._penalty([-2.0, 3.0], [-1.0], [1.0, 1.0], [2.0]) == 3.0 + 2.0

    def test_signed_rewards_satisfied_constraints(self):
        p = self._min_problem("signed")
        assert p._penalty([-2.0, 3.0], [-1.0], [1.0, 1.0], [2.0]) == 1.0 - 2.0

    def test_solver_sets_sign_on_min_problem(self):
        s = _solver(TNK(), penalty_sign="signed")
        assert s.min_problem.penalty_sign == "signed"

    def test_signed_changes_lagrangian(self):
        s = _solver(TNK(), penalty_sign="signed")
        s.min_problem.set_objective_multipliers([([1.0, 1.0], []), ([1.0, 1.0], [])])
        e = s.min_problem._lagrangian(
            Evaluation(fitness=[0.5, 0.5], constraints_inequality=[-1.0, -1.0]))
        assert e.fitness == [-1.5, -1.5]


class TestRescore:
    def _count_rescores(self, **kwargs):
        s = _solver(TNK(), **kwargs)
        calls = []
        original = s.objective_solver.rescore
        s.objective_solver.rescore = lambda *a, **k: (calls.append(1), original(*a, **k))
        _run(s, 15)
        return len(calls)

    def test_rescore_on_by_default(self):
        assert self._count_rescores() > 0

    def test_rescore_off(self):
        assert self._count_rescores(rescore=False) == 0

    def test_rescore_off_shared(self):
        assert self._count_rescores(rescore=False, multiplier_mode="shared") == 0


def test_objective_multipliers_per_objective():
    s = _run(_solver(TNK()), 5)
    pairs = s.objective_multipliers
    assert len(pairs) == 2
    for (mu, la), solver in zip(pairs, s.multiplier_solvers):
        assert len(mu) == 2 and la == []
        assert all(m >= 0.0 for m in mu)
