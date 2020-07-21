#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the Population class in the particlefilter package. """
import pytest
import numpy as np
import os

from particlefilter import Population


def get_population_and_data(n=10, dim=2):
    x = np.random.rand(n, dim)
    y = np.sum(x**2, axis=1)
    pop = Population(x, y)

    return x, y, pop


def test_input_data_validation():
    x = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        _ = Population(x, np.random.rand(9))
    with pytest.raises(TypeError):
        _ = Population(x, None)
    with pytest.raises(TypeError):
        _ = Population(None, np.random.rand(9))
    with pytest.raises(ValueError):
        _ = Population(x, np.random.rand(10, 2))


def test_origin():
    n = 10
    x, y, pop = get_population_and_data(n)

    assert len(pop.origin_iteration) == n
    assert np.sum(pop.origin_iteration) == 0
    assert pop._now == 1

    pop.append(x, y)
    assert len(pop.origin_iteration) == 2*n
    assert np.sum(pop.origin_iteration) == n
    pop._now += 1

    pop.append(x, y)
    assert len(pop.origin_iteration) == 3*n
    assert np.sum(pop.origin_iteration) == 3*n
    pop._now += 1
    assert pop._now == 3


def test_kill():
    n = 10
    x, y, pop = get_population_and_data(n)
    pop.append(x, y)

    pop.set_kill_list(pop.origin_iteration == 0)
    pop.end_iteration()

    assert len(pop) == n
    assert len(pop.origin_iteration) == n
    assert np.sum(pop.origin_iteration) == n


def test_procreation_rates_validation():
    n = 10
    x, y, pop = get_population_and_data(n)

    with pytest.raises(ValueError):
        pop.set_procreation_rates(np.zeros(9))

    with pytest.raises(ValueError):
        pop.set_procreation_rates(np.ones(10)*0.9)

    with pytest.raises(ValueError):
        rates = np.ones(10)
        rates[5] = 0.999
        pop.set_procreation_rates(rates)

    pop.set_procreation_rates(np.ones(10))


def test_function_extremes():
    x, y, pop = get_population_and_data()

    a, b = pop.get_function_extremes()
    assert np.amin(y) == a
    assert np.amax(y) == b


def test_latest_data():
    n = 10
    x, y, pop = get_population_and_data()

    a, b = np.random.rand(n, 2), np.random.rand(n, 1)
    pop.append(a, b)
    c, d = pop.get_latest_data()

    assert np.array_equal(a, c)
    assert np.array_equal(b, d)


def test_data_on_kill_list():
    x, y, pop = get_population_and_data()

    pop.append(x+1, y+1)
    pop.set_kill_list(pop.origin_iteration == 1.0)

    a, b = pop.get_data_on_kill_list()
    assert np.array_equal(a, x+1)
    assert np.array_equal(b, y.reshape(-1, 1)+1)


def test_get_procreating_data():
    x, y, pop = get_population_and_data()

    pop.append(x+1, y+1)
    pop.set_procreation_rates(pop.origin_iteration)

    a, b = pop.get_procreating_data()
    print(a)
    print(pop.x)
    assert np.array_equal(a, x+1)
    assert np.array_equal(b, y.reshape(-1, 1)+1)


def test_reset_nonzero():
    x, y, pop = get_population_and_data()

    assert pop
    pop.reset()
    assert not pop


def test_stdevs():
    n = 10
    x = np.random.rand(n, 2)
    y = np.arange(n)

    pop = Population(x, y)
    rates = 1.0*(np.random.rand(n) > 0.6)
    pop.set_procreation_rates(rates)

    pop.set_stdevs(np.ones(n), False)
    assert pop.stdevs.shape == x.shape
    pop.set_stdevs(np.ones(int(np.sum(rates))), True)
    assert pop.stdevs.shape == x.shape
    pop.set_stdevs(np.random.rand(n, 2), False)
    assert pop.stdevs.shape == x.shape
    pop.set_stdevs(np.random.rand(int(np.sum(rates)), 2), True)
    assert pop.stdevs.shape == x.shape


def test_get_indices_from_rates():
    x, y, pop = get_population_and_data(5)

    pop.set_procreation_rates(np.array([0, 1, 2, 3, 1]))
    ind = pop._get_indices_from_rates()
    assert np.array_equal(ind, np.array([1, 2, 2, 3, 3, 3, 4]))


def test_sample():
    n = 5000
    dim = 1
    x, y, pop = get_population_and_data(n, dim)

    covs = np.zeros((n, dim, dim))
    for i in range(dim):
        covs[:, i, i] = 0.001
    _ = pop._sample(x, covs)


def test_out_of_bounds():
    _, _, pop = get_population_and_data()

    x = np.array([[1]])
    assert pop._is_out_of_bounds(x, np.array([[-2, -1]])) == [True]
    assert pop._is_out_of_bounds(x, np.array([[0, 2]])) == [False]
    assert pop._is_out_of_bounds(x, np.array([[3, 5]])) == [True]

    x = np.array([[1, 1], [2, 2], [3, 3]])
    assert np.array_equal(pop._is_out_of_bounds(x, None),
                          [False, False, False])
    oob = pop._is_out_of_bounds(x, np.array([[-2, -1], [-2, -1]]))
    assert np.array_equal(oob, [True, True, True])
    bounds = np.array([[0, 2.4], [1.2, 3.1]])
    oob = pop._is_out_of_bounds(x, bounds)
    assert np.array_equal(oob, [True, False, True])


def test_procreate():
    x, y, pop = get_population_and_data(5)
    pop.set_procreation_rates(np.array([0, 1, 1, 0, 2]))
    pop.set_stdevs(np.ones((5, 2))*0.5, False)

    _ = pop.procreate(np.array([[0.2, 0.7], [0.1, 0.6]]))


def test_stringify():
    x, y, pop = get_population_and_data(5)
    print(pop)
    s = pop.__str__()
    assert 'Population' in s
    assert 'datapoints' in s
    assert 'iteration(s)' in s


def test_mean_stdev_sampling_errors():
    x, y, pop = get_population_and_data(5)
    pop.reset()
    with pytest.raises(Exception):
        pop.get_means_and_stdevs_for_sampling()

    x, y, pop = get_population_and_data(5)
    pop.set_procreation_rates(np.arange(5))
    with pytest.raises(Exception):
        pop.get_means_and_stdevs_for_sampling()


def test_validate_input_data():
    x, y, pop = get_population_and_data(5)
    pop.validate_input_data(np.random.rand(5, 2), np.random.rand(5))
    with pytest.raises(TypeError):
        pop.validate_input_data(1, np.random.rand(5))
    with pytest.raises(TypeError):
        pop.validate_input_data(np.random.rand(5), 1)
    with pytest.raises(ValueError):
        pop.validate_input_data(np.random.rand(5), np.random.rand(5))
    with pytest.raises(ValueError):
        pop.validate_input_data(np.random.rand(5, 2), np.random.rand(5, 4, 3))
    with pytest.raises(ValueError):
        pop.validate_input_data(np.random.rand(5, 2), np.random.rand(3))


def test_procreation_rates_errors():
    x, y, pop = get_population_and_data(5)
    with pytest.raises(TypeError):
        pop.validate_procreation_rates(1)
    with pytest.raises(ValueError):
        pop.validate_procreation_rates(np.random.rand(4))
    with pytest.raises(ValueError):
        pop.validate_procreation_rates(np.random.rand(5))
    pop.validate_procreation_rates(np.arange(5))


def test_stdevs_errors():
    x, y, pop = get_population_and_data(5)
    pop.set_procreation_rates(np.array([0, 0, 1, 1, 1]))
    with pytest.raises(TypeError):
        pop.validate_stdevs(1)
    with pytest.raises(ValueError):
        pop.validate_stdevs(np.random.rand(3), False)
    with pytest.raises(ValueError):
        pop.validate_stdevs(np.random.rand(5))
    pop.validate_stdevs(np.arange(5), False)
    pop.validate_stdevs(np.arange(3), True)


def test_kill_list_errors():
    x, y, pop = get_population_and_data(5)
    with pytest.raises(Exception):
        pop.validate_kill_list(1)
    with pytest.raises(Exception):
        pop.validate_kill_list(np.arange(4) < 3)
    kl = pop.validate_kill_list([True, True, False, False, True])
    kl = pop.validate_kill_list(np.arange(5).reshape(5, 1, 1) < 3)
    assert kl.shape == (5, )


def test_double_seeding():
    x, y, pop = get_population_and_data(5)
    with pytest.raises(Exception):
        pop.set_seed(x, y)


def test_append_wrong_shapes():
    x, y, pop = get_population_and_data(5)
    with pytest.raises(ValueError):
        pop.append(np.random.rand(5, 4), y)
    pop.append(x, np.random.rand(5, ))
    pop.append(x, y)


def test_procreate_boundary_errors():
    x, y, pop = get_population_and_data(5)
    with pytest.raises(Exception):
        pop.procreate()
    with pytest.raises(TypeError):
        pop.procreate(1)

    shape = x.shape[1:] + (2, )
    boundaries = np.random.rand(*shape)
    boundaries[..., 0] = 0
    boundaries[..., 1] = 1
    with pytest.raises(Exception):
        pop.procreate(boundaries)

    shape = (3, 2)
    boundaries = np.random.rand(*shape)
    boundaries[..., 0] = 0
    boundaries[..., 1] = 1
    with pytest.raises(Exception):
        pop.procreate(boundaries)


def test_procreate_max_attempts():
    x, y, pop = get_population_and_data(5)
    pop.set_procreation_rates(np.ones(5))
    pop.set_stdevs(np.ones(5)*0.0000001)
    samples = pop.procreate(boundaries=np.array([[10, 11], [10, 11]]))
    assert len(samples) == 0


def test_save_1(tmp_path):
    x, y, pop = get_population_and_data(5)
    filepath = '{}/csv_file.csv'.format(tmp_path)
    pop.save(filepath)
    assert os.path.exists(filepath)


def test_save_2(tmp_path):
    x, y, pop = get_population_and_data(5)
    pop.set_stdevs(np.random.rand(5), False)
    filepath = '{}/csv_file.csv'.format(tmp_path)
    pop.save(filepath)
    assert os.path.exists(filepath)


def test_graveyard(tmp_path):
    x, y, pop = get_population_and_data(10)

    assert not pop.has_graveyard()
    path = str(tmp_path)+"/graveyard.csv"
    # Make sure file is made by make_graveyard
    assert not os.path.exists(path)
    pop.make_graveyard(path)
    assert os.path.exists(path)
    assert pop.has_graveyard()

    # Populate graveyard
    pop.append(x+1, y+1)
    pop._now += 1
    pop.append(x+2, y+2)
    pop.set_kill_list(pop.origin_iteration == 0.0)
    pop.send_to_graveyard(*pop.get_data_on_kill_list_with_origin())

    # Assert that there are 10 datapoints on graveyard
    graveyard = np.genfromtxt(path, skip_header=1, delimiter=',')
    assert graveyard.shape == (10, 5)

    # Set new graveyard
    assert pop._graveyard_handle is not None
    pop.make_graveyard()
    assert pop._graveyard_handle is None


def test_empty_graveyard(tmp_path):
    x, y, pop = get_population_and_data(10)
    path = str(tmp_path)+"/graveyard.csv"
    pop.make_graveyard(path)

    origin = np.zeros(10).reshape(10, 1)
    pop.send_to_graveyard(x, y, origin)
    graveyard = np.genfromtxt(path, skip_header=1, delimiter=',')
    assert graveyard.shape == (10, 5)

    pop.make_graveyard()
    pop.send_to_graveyard(x, y, origin)
    graveyard = np.genfromtxt(path, skip_header=1, delimiter=',')
    assert graveyard.shape == (10, 5)
