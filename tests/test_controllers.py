#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the controller module. in the particlefilter package """
import pytest
import numpy as np

from particlefilter.controllers import (get_width_controller,
                                        get_stdev_controller,
                                        get_kill_controller)


def test_width_controller_creation():
    # Get a controller
    func = get_width_controller()
    assert callable(func)


def test_width_controller_defaults():
    initial_width = 10
    width = initial_width
    # Get a controller
    func = get_width_controller()
    # Loop over iterations and check if calculated with matches expectatation
    for i in range(10):
        assert width*0.95 == func(i, width)
        width = width*0.95


def test_width_controller_custrom_rate():
    initial_width = 10
    width = initial_width
    rate = 0.5
    # Get a controller
    func = get_width_controller(rate)
    # Loop over iterations and check if calculated with matches expectatation
    for i in range(10):
        assert width*rate == func(i, width)
        width = width*rate


def test_width_controller_custrom_n():
    initial_width = 10
    width = initial_width
    rate = 0.95
    n = 2
    # Get a controller
    func = get_width_controller(rate, n)
    # Loop over iterations and check if calculated with matches expectatation
    for i in range(10):
        new_width = width
        if i % n == 0:
            new_width = new_width * rate
        assert new_width == func(i, width)
        width = new_width


def test_width_controller_errors():
    with pytest.raises(ValueError):
        _ = get_width_controller(1.1)
    with pytest.raises(ValueError):
        _ = get_width_controller(-0.1)
    with pytest.raises(ValueError):
        _ = get_width_controller(apply_every_n_iterations=1.1)


def test_kill_controller_creation():
    func = get_kill_controller()
    assert callable(func)


def test_kill_controller_errors():
    with pytest.raises(ValueError):
        _ = get_kill_controller(survival_rate=-0.1)
    with pytest.raises(ValueError):
        _ = get_kill_controller(survival_rate=1.1)


def test_kill_controller_defaults():
    func = get_kill_controller()

    y = np.arange(10)
    is_new = np.array([True, True, False, False, False,
                       False, False, False, False, False])
    iteration_size = 5
    expectation = np.array([False, False, False, False, True,
                            True, True, True, True, True])
    assert np.array_equal(func(y, is_new, iteration_size), expectation)


def test_kill_controller_survivalrate():
    func = get_kill_controller(survival_rate=0.5)

    y = np.arange(10)
    is_new = np.array([True, True, False, False, False,
                       False, False, False, False, False])
    iteration_size = 5
    expectation = np.array([False, False, False, False, False,
                            False, True, True, True, True])
    reality = func(y, is_new, iteration_size)
    print(reality)
    assert np.array_equal(reality, expectation)


def test_kill_controller_defaults_complicated():
    func = get_kill_controller()

    y = np.arange(10)
    is_new = np.array([False, False, False, False, False,
                       False, False, False, True, True])
    iteration_size = 5
    expectation = np.array([False, False, True, True, True,
                            True, True, True, False, False])
    assert np.array_equal(func(y, is_new, iteration_size), expectation)


def test_kill_controller_iterationsize():
    func = get_kill_controller(cut_to_iteration_size=True)

    y = np.arange(10)
    is_new = np.array([False, False, False, False, False,
                       False, False, False, True, True])
    iteration_size = 2
    expectation = np.array([False, False, True, True, True,
                            True, True, True, True, True])
    reality = func(y, is_new, iteration_size)
    assert np.array_equal(reality, expectation)


def test_stdev_controller_creation():
    func = get_stdev_controller()
    assert callable(func)


def test_stdev_controller_errors():
    func = get_stdev_controller(2, 10)
    # min max exchanged check
    with pytest.raises(ValueError):
        _ = get_stdev_controller(1, -1)
    # Boundary type check
    width = 0.1
    boundaries = 1
    x = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        _ = func(width, boundaries, x)
    # Boundary shape check
    boundaries = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        _ = func(width, boundaries, x)


def test_stdev_controller_defaults():
    func = get_stdev_controller()

    width = 0.1
    boundaries = np.array([[-1, 1], [0, 20]])
    x = np.random.rand(10, 2)

    expectation = np.ones(x.shape)
    expectation[:, 0] = 0.2
    expectation[:, 1] = 2.0
    reality = func(width, boundaries, x)
    assert np.array_equal(expectation, reality)


def test_stdev_controller_none_boundaries():
    func = get_stdev_controller(inf_replace=10)

    width = 0.1
    boundaries = None
    x = np.random.rand(10, 2)

    expectation = np.ones(x.shape)*2.0
    reality = func(width, boundaries, x)
    assert np.array_equal(expectation, reality)


def test_stdev_controller_inf_boundaries():
    func = get_stdev_controller(inf_replace=10)

    width = 0.1
    boundaries = np.array([[-1, 1], [-np.inf, np.inf]])
    x = np.random.rand(10, 2)

    expectation = np.ones(x.shape)
    expectation[:, 0] = 0.2
    expectation[:, 1] = 2.0
    reality = func(width, boundaries, x)
    assert np.array_equal(expectation, reality)


def test_stdev_controller_logarithmic():
    func = get_stdev_controller(logarithmic=True)

    width = 0.1
    boundaries = np.array([[-1, 1], [-10, 10]])
    x = np.arange(20).reshape(10, 2)

    expectation = np.ones(x.shape)
    expectation[:, 0] = 0.2
    expectation[:, 1] = 2.0
    expectation *= x
    reality = func(width, boundaries, x)
    assert np.array_equal(expectation, reality)


def test_stdev_controller_minmax():
    func = get_stdev_controller(min_stdev=1, max_stdev=5)

    width = 0.1
    boundaries = np.array([[-1, 1], [-np.inf, np.inf]])
    x = np.random.rand(10, 2)

    expectation = np.ones(x.shape)
    expectation[:, 0] = 1.0
    expectation[:, 1] = 5.0
    reality = func(width, boundaries, x)
    assert np.array_equal(expectation, reality)
