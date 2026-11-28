# test/problem/test_dtnk_dimension.py
"""Tests for the DTNK dimension parameter and set_environment."""

import math
import random

import numpy as np
import pytest

from cilpy.problem.dynamic_multi_objective import DTNK, DTNK4, DTNK5, DTNK6, FDA1

PROBLEMS = [DTNK, DTNK4, DTNK5, DTNK6]


@pytest.mark.parametrize("cls", PROBLEMS)
def test_default_dimension_is_two(cls):
    p = cls()
    assert p.dimension == 2
    assert p.bounds == ([0.0, 0.0], [math.pi, math.pi])


@pytest.mark.parametrize("cls", PROBLEMS)
def test_bounds_for_dimension_ten(cls):
    p = cls(dimension=10)
    assert p.dimension == 10
    assert p.bounds == ([0.0] * 10, [math.pi] * 10)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_dimension_below_two_raises(cls):
    with pytest.raises(ValueError):
        cls(dimension=1)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_distance_variables_at_optimum_give_2d_values(cls):
    random.seed(3)
    p2, p10 = cls(tau_t=1, n_t=5), cls(tau_t=1, n_t=5, dimension=10)
    for k in (0, 3, 7):
        p2.set_environment(k)
        p10.set_environment(k)
        for _ in range(20):
            x = [random.uniform(0, math.pi), random.uniform(0, math.pi)]
            a = p2.evaluate(x)
            b = p10.evaluate(x + [math.pi / 2] * 8)
            assert a.fitness == b.fitness
            assert a.constraints_inequality == b.constraints_inequality


@pytest.mark.parametrize("cls", PROBLEMS)
def test_distance_variables_scale_the_position(cls):
    p2, p10 = cls(), cls(dimension=10)
    x = [0.4, 0.7]
    extra = [math.pi / 2] * 7 + [math.pi / 2 + 0.5]
    g = 1.0 + 0.25
    assert p10.evaluate(x + extra).fitness == pytest.approx(
        p2.evaluate([g * 0.4, g * 0.7]).fitness)
    assert p10.evaluate(x + extra).constraints_inequality == pytest.approx(
        p2.evaluate([g * 0.4, g * 0.7]).constraints_inequality)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_front_does_not_depend_on_dimension(cls):
    p2, p10 = cls(tau_t=1, n_t=5), cls(tau_t=1, n_t=5, dimension=10)
    for k in (0, 4):
        p2.set_environment(k)
        p10.set_environment(k)
        assert np.array_equal(p2.true_pareto_front(100), p10.true_pareto_front(100))


class TestSetEnvironment:
    def test_matches_begin_iteration(self):
        a, b = FDA1(tau_t=7, n_t=4), FDA1(tau_t=7, n_t=4)
        for _ in range(7 * 3):
            a.begin_iteration()
        b.set_environment(3)
        assert a.t == b.t == pytest.approx(0.75)
        assert a.environment == b.environment == 3

    def test_dtnk4_uses_environment_index(self):
        a, b = DTNK4(tau_t=5), DTNK4(tau_t=5)
        for _ in range(5 * 6):
            a.begin_iteration()
        b.set_environment(6)
        x = [1.2, 2.1]
        assert a.evaluate(x).constraints_inequality == b.evaluate(x).constraints_inequality

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            FDA1().set_environment(-1)

    def test_iterations_continue_from_set_environment(self):
        p = DTNK(tau_t=10, n_t=10)
        p.set_environment(2)
        for _ in range(10):
            p.begin_iteration()
        assert p.environment == 3
