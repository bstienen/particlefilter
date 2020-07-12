# -*- coding: utf-8 -*-
import logging
import numpy as np

logger = logging.getLogger(__name__)


def validate_boundaries(boundaries, dimensionality=None):
    """ Checks if the boundaries that the user can provide to the
    `procreate` method follow the expected rules.

    Allowed settings of the provided boundaries are:
    - `None`, indicating that there are no boundaries. This value will be
      returned unaltered.
    - A `numpy.ndarray` of shape `(1, 2)` (or an array that is reshapable
      to this shape) that provides the upper and lower boundary that are
      the same for all dimensions. This method will expand this array to
      `(nDimensions, 2)`, where all rows will have the same entries, making
      the boundary array more explicit.
    - A `numpy.ndarray` of shape `(nDimensions, 2)` indicating for each
      of the dimensions the upper and lower boundary.

    Args:
        boundaries: `None` or `numpy.ndarray`. See text above for allowed
            shapes for the `numpy.ndarray`.
        dimensionality: If set to `None` (as is the default), the only
            requirement on `boundaries` that will be checked is that it has
            shape (..., 2). If set as `int` or `tuple`, it is explicitly
            checked and reshaped (if possible) to `(dimensionality, 2)` or
            `(*dimensionality, 2)` respectively.

    Returns:
        `None` or a `numpy.ndarray` of shape `(nDimensions, 2)` containing
        explicitly the boundaries for each of the dimensions. See the full
        docstring for how the input is transformed to form this output.

    Raises:
        TypeError: The provided boundaries should be `None` or of type
            `numpy.ndarray`. Provided as a(n) (?).
        ValueError: The last axis of the `boundaries` array should define the
            lower and upper boundary in each of the dimensions, and as such
            have size 2. Provided array had shape (?).
        TypeError: Dimensionality should be provided as an `int` or `tuple`.
            Provided was (?), a(n) (?).
        ValueError: Boundaries for sampling should be provided as either `None`
            or a `numpy.ndarray` of shape `(2,)`, `(1, 2)` or `(x, 2)`, with
            `x` equaling the requested dimensionality through the
            `dimensionality` argument. Provided was an array of shape {}.
        ValueError: Boundaries for sampling provided as a `numpy.ndarray` with
            shape `(..., 2)` should have all its lower bounds lower than its
            upper bounds. There were {} dimensions for which this was not the
            case."""
    # Check type of `boundaries`
    if boundaries is not None and not isinstance(boundaries, np.ndarray):
        raise TypeError("The provided boundaries should be `None` or of "
                        "type `numpy.ndarray`. Provided as "
                        "a(n) {}.".format(type(boundaries)))
    # If boundaries is `None`, return it. `None` is interpreted in the rest
    # of the package as the absence of boundaries (i.e. [-np.inf, np.inf] for
    # all dimensions)
    if boundaries is None:
        return None
    # Regardless of the dimensionality: the last axis of the boundaries should
    # be of size 2 if it is a `numpy.ndarray`. let's check this.
    if boundaries.shape[-1] != 2:
        raise ValueError("The last axis of the `boundaries` array should "
                         "define the lower and upper boundary in each of "
                         "the dimensions, and as such have size 2. Provided "
                         "array had shape {}.".format(boundaries.shape))
    # If `dimensionality` was set, we check if the `boundaries` array matches
    # this dimensionality and if not, whether it can be recast to this shape.
    # If not, raise a value error.
    reshape = False
    if dimensionality is not None:
        # Determine required shape
        requested_shape = (0, )
        if isinstance(dimensionality, int):
            requested_shape = (dimensionality, 2)
        elif isinstance(dimensionality, tuple):
            requested_shape = (dimensionality + (2, ))
        else:
            raise TypeError("Dimensionality should be provided as an `int` or "
                            "`tuple`. Provided was '{}', a(n) {}.".format(
                                dimensionality, type(dimensionality)))
        # Check boundaries shape
        if boundaries.shape != requested_shape:
            reshape = True
            try:
                original_shape = boundaries.shape
                boundaries = boundaries.reshape((1, 2))
            except ValueError:
                raise ValueError("Boundaries for sampling should be provided "
                                 "as either `None` or a `numpy.ndarray` of "
                                 "shape `(2,)`, `(1, 2)` or `(x, 2)`, with "
                                 "`x` equaling the requested dimensionality "
                                 "through the `dimensionality` argument. "
                                 "Provided was an array of shape {} and the "
                                 "requested dimensionality was {}.".format(
                                     original_shape, dimensionality))
    # Let's also perform a sanity check on whether the lower boundaries are
    # indeed all lower than the upper boundaries.
    difference = boundaries[..., 1] - boundaries[..., 0]
    n_not_positive = np.sum(difference <= 0)
    if n_not_positive:
        raise ValueError("Boundaries for sampling provided as a "
                         "`numpy.ndarray` with shape `(..., 2)` should have "
                         "all its lower bounds lower than its upper bounds. "
                         "There were {} dimensions for which this was not the "
                         "case.".format(n_not_positive))
    # Return the boundaries, but reshape them to requested_shape if this
    # is necesary
    if reshape:
        boundaries = boundaries * np.ones(requested_shape)
    return boundaries
