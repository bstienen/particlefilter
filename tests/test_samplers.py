#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the samplers module in the particlefilter package. """
import pytest
import numpy as np

from particlefilter.samplers import uniform_sampler


def test_uniform_sampler():
    samples = uniform_sampler(20, np.array([[1.0, 2.0]]))
    print(samples)
    assert samples.shape == (20, 1)
    assert np.amin(samples) >= 1.0
    assert np.amax(samples) <= 2.0

    samples = uniform_sampler(20, np.array([[0.0, 3.0],
                                            [1.0, 10.0],
                                            [9.0, 111000.0]]))
    assert samples.shape == (20, 3)

    boundaries = np.random.rand(13, 17, 2)
    boundaries[..., 1] = boundaries[..., 1] + 2
    samples = uniform_sampler(20, boundaries)
    assert samples.shape == (20, 13, 17)


def test_unform_sampler_errors():
    with pytest.raises(TypeError):
        _ = uniform_sampler(20, None)
