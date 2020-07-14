#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the ParticleFilter class in the particlefilter package. """
import pytest
import numpy as np

from particlefilter import ParticleFilter
from particlefilter.samplers import uniform_sampler
from particlefilter.controllers import get_kill_controller


def func(x):
    return np.sum(x**2, axis=1)


def callback(iteration, width, function, data):
    raise NotImplementedError


def incorrect_callback(iteration, width, function):
    pass


def test_creation():
    pf = ParticleFilter(func, 100)
    pf.validate_emptiness()


def test_creation_errors():
    with pytest.raises(TypeError):
        _ = ParticleFilter()
    with pytest.raises(TypeError):
        _ = ParticleFilter(func([1]), 100)
    with pytest.raises(TypeError):
        _ = ParticleFilter(func, 100, boundaries=20)


def test_creation_width():
    pf = ParticleFilter(func, 100, initial_width=1.0)
    assert pf.initial_width == 1.0
    assert pf.width == 1.0


def test_callbacks():
    pf = ParticleFilter(func, 100)
    pf.add_callback('at_start_of_iteration', callback)
    assert pf._callbacks['at_start_of_iteration'] == [callback]
    pf.add_callback('at_start_of_iteration')
    assert pf._callbacks['at_start_of_iteration'] == []


def test_callback_errors():
    pf = ParticleFilter(func, 100)
    with pytest.raises(TypeError):
        pf.add_callback("at_start_of_iteration", "callback")
    with pytest.raises(ValueError):
        pf.add_callback("at_start_of_iteration", incorrect_callback)
    with pytest.raises(ValueError):
        pf.add_callback("at_some_undefined_point", callback)


def test_callback_call():
    pf = ParticleFilter(func, 100)
    pf.add_callback("at_start_of_run", callback)
    with pytest.raises(NotImplementedError):
        pf.callback("at_start_of_run")
    pf.add_callback("at_start_of_run")
    pf.callback("at_start_of_run")


def test_sampler():
    pf = ParticleFilter(func, 100, boundaries=np.array([[0, 1], [0, 1]]))
    pf.validate_emptiness()
    pf.sample_seed(10, uniform_sampler)
    with pytest.raises(Exception):
        pf.validate_emptiness()


def test_sampler_errors():
    pf = ParticleFilter(func, 100, boundaries=np.array([[0, 1], [0, 1]]))
    with pytest.raises(TypeError):
        pf.sample_seed(10, "uniform_sampler")
    with pytest.raises(ValueError):
        pf.sample_seed(10, callback)


def test_function_errors():
    _ = ParticleFilter(func, 100)
    with pytest.raises(TypeError):
        _ = ParticleFilter("func", 100)
    with pytest.raises(ValueError):
        _ = ParticleFilter(callback, 100)


def test_initialise_run():
    pf = ParticleFilter(func, 100)
    pf.add_callback("at_start_of_run", callback)
    pf.iteration = 100
    assert pf.iteration == 100
    with pytest.raises(NotImplementedError):
        pf.initialise_run()
    assert pf.iteration == 0


def test_calculate_procreation_rates_1():
    pf = ParticleFilter(func, 100)

    x = np.random.rand(10, 2)
    y = np.arange(10)
    pf.set_seed(x, y)

    rates = pf._calculate_procreation_rates(y, 10000)
    rates[:-1] = rates[:-1] - rates[1:]
    positive = np.sum(rates[:-1] > 0)
    assert positive == len(y)-1


def test_calculate_procreation_rates_2():
    pf = ParticleFilter(func, 100)

    x = np.random.rand(10, 2)
    y = np.ones(10)
    pf.set_seed(x, y)

    rates = pf._calculate_procreation_rates(y, 10000)
    assert len(np.unique(rates)) == 1
    assert rates[0] > 0

    rates = pf._calculate_procreation_rates(y, 13)
    assert len(np.unique(rates)) == 2
    assert np.amax(rates) - np.amin(rates) == 1
    assert np.sum(rates) == 13


def test_run_iteration():
    pf = ParticleFilter(func, 10,
                        initial_width=0.00001,
                        boundaries=np.array([[0, 1], [0, 1]]),
                        kill_controller=get_kill_controller(0.2, False))

    x = np.random.rand(100, 2)
    y = np.arange(100)
    pf.set_seed(x, y)

    pf.initialise_run()
    pf.run_iteration()
