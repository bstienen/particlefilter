# -*- coding: utf-8 -*-
import logging
import numpy as np

from .utils import validate_boundaries

logger = logging.getLogger(__name__)


def uniform_sampler(n, boundaries):
    """ Sampler function that samples `n` samples uniform in the ranges
    specified by the `boundaries` argument.

    Boundaries are expected to be a `numpy.ndarray` and be valid according
    to the definition in `utils.validate_boundaries`. Note that this is a
    stronger requirement, as the method in `utils` also accepts `None` as a
    valid boundary definition.

    Args:
        n: Integer indicating how many samples need to be taken.
        boundaries: `numpy.ndarray` containing the upper and lower boundaries
            within which the samples need to be taken. The array should be
            valid according to the rules of `utils.validate_boundaries`.

    Returns:
        A `numpy.ndarray` containing uniform samples taken in the region
        defined by `boundaries`.

    Raises:
        TypeError: `uniform_sampler` expects boundaries to be a
            `numpy.ndarray`. Provided was '(?)', a(n) (?). """
    # Check boundaries
    if boundaries is None:
        raise TypeError("Samplers expect boundaries to be a `numpy.ndarray`. "
                        "Provided was '{}', a(n) {}.".format(
                            boundaries, type(boundaries)))
    boundaries = validate_boundaries(boundaries)
    # Get minima and maxima from boundaries
    minima = boundaries[..., 0]
    maxima = boundaries[..., 1]
    # Sample `n` datapoints in the shape of boundaries.shape[:-1]
    shape = (n, ) + boundaries.shape[:-1]
    samples = np.random.rand(*shape)
    # Apply range and shift to them to get the samples requested.
    samples = samples * (maxima - minima)
    samples = samples + minima
    return samples


def log_sampler(n, boundaries):
    raise NotImplementedError


def get_composite_sampler(distributions):
    raise NotImplementedError

    def sampler(n, boundaries):
        raise NotImplementedError
    return sampler
