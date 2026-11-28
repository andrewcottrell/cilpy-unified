# test/solver/test_sentry_refresh.py
"""Tests that a change detected by the archive sentries also refreshes the
fixed sentry."""
import random

import numpy as np

from cilpy.problem.dynamic_multi_objective import DTNK
from cilpy.solver.mgpso import MGPSO


def _solver(problem):
    random.seed(2)
    np.random.seed(2)
    return MGPSO(name="t", problem=problem, swarm_size=20, w=0.72, c1=1.49,
                 c2=1.49, c3=1.49, sentry_mode="archive")


def _empty_archive(solver):
    solver._archive.positions = []
    solver._archive.evaluations = []
    solver._archive._invalidate()


def test_change_is_not_detected_twice_when_archive_empties():
    problem = DTNK(tau_t=1000, n_t=10)
    s = _solver(problem)
    for _ in range(5):
        s.step()
    assert len(s._archive) > 0

    problem.set_environment(1)
    s.step()
    assert s.n_changes_detected == 1

    _empty_archive(s)
    s.step()
    assert s.n_changes_detected == 1


def test_fixed_sentry_still_detects_next_change():
    problem = DTNK(tau_t=1000, n_t=10)
    s = _solver(problem)
    for _ in range(5):
        s.step()
    problem.set_environment(1)
    s.step()
    _empty_archive(s)
    problem.set_environment(2)
    s.step()
    assert s.n_changes_detected == 2
