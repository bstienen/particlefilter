#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Tests for the utils module in the particlefilter package. """
import pytest
import numpy as np

from particlefilter.utils import validate_boundaries


def test_validate_boundaries():
    b = validate_boundaries(None)
    assert b is None

    b = validate_boundaries(np.array([[1, 2]]))
    assert b.shape == (1, 2)

    bounds = np.arange(8).reshape(4, 2)
    b = validate_boundaries(bounds)
    assert b.shape == (4, 2)

    with pytest.raises(ValueError):
        bounds = np.arange(8)[::-1].reshape(4, 2)
        b = validate_boundaries(bounds)


def test_validate_boundaries_dimensionality():
    b = validate_boundaries(None, 4)
    assert b is None

    b = validate_boundaries(np.array([[1, 2]]), 4)
    assert b.shape == (4, 2)

    b = validate_boundaries(np.array([[1, 2]]), (4, 3))
    assert b.shape == (4, 3, 2)


def test_validate_boundaries_errors():
    with pytest.raises(TypeError):
        _ = validate_boundaries(1)
    with pytest.raises(ValueError):
        _ = validate_boundaries(np.random.rand(3, 4))
    with pytest.raises(TypeError):
        _ = validate_boundaries(np.random.rand(1, 2), "error")
    with pytest.raises(ValueError):
        _ = validate_boundaries(np.random.rand(5, 2), 6)
    boundaries = np.arange(8).reshape(4, 2)
    _ = validate_boundaries(boundaries, 4)
    boundaries = np.arange(24).reshape(4, 3, 2)
    _ = validate_boundaries(boundaries, (4, 3))
