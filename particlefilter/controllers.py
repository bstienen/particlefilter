# -*- coding: utf-8 -*-
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_width_controller(decay_rate=0.95, apply_every_n_iterations=1):
    """ Get a simple width controller that exponentially decreases the width
    every nth iteration.

    Args:
        decay_rate: `float` indicating the factor with which the width needs
            to be multiplied every nth iteration. Its value should lie
            between 0.0 and 1.0. Default is 0.95.
        apply_every_n_iterations: `int` defining at which iterations the
            current width should be multiplied by the decay rate. Default is 1.

    Returns:
        A controller function that takes two arguments (the current iteration
        and the current width) and returns the new width as a float. See its
        docstring for more information.

    Raises:
        ValueError: The decay_rate should be a value between 0.0 and 1.0, but
            provided was (?).
        ValueError: `apply_every_n_iterations` expects an integer, but
            provided was (?)."""
    # Check if decay_rate and apply_every_n_iterations are valid
    if not 0.0 <= decay_rate <= 1.0:
        raise ValueError("The decay_rate should be a value between 0.0 and "
                         "1.0, but provided was {}.".format(decay_rate))
    if int(apply_every_n_iterations) - apply_every_n_iterations != 0.0:
        raise ValueError("`apply_every_n_iterations` expects an integer, but "
                         "provided was {}.".format(apply_every_n_iterations))

    # Define controller
    def controller(iteration, width):
        """ Function that can alter the width parameter of a Particle Filter.
        It is configured using the parent function `get_width_controller`.

        This specific controller multiplies the current width by the
        `decay_rate` if the current iteration is a multiple of the
        `apply_every_n_iterations` variable.

        Args:
            iteration: `int` defining the iteration in which the Particle
                Filter is currently running. 0 indicates the initial sampling,
                after that the iteration increases by 1 with each iteration.
            width: The current width as `float`.

        Returns:
            New width as `float`. """
        if iteration % int(apply_every_n_iterations) == 0:
            width *= decay_rate
        return width

    return controller


def get_kill_controller(survival_rate=0.2, cut_to_iteration_size=False):
    """ Get a simple kill controller that kills all datapoints that are (1) not
    sampled in the most recent iteration and that (2) are not part of the
    `survival_rate` fraction of old points with the best (i.e. lowest)
    function values.

    By setting the `cut_to_iteration_size` argument to `True`, the resulting
    selection is cut down to match the iteration size in length (which avoids
    the exponential growth of the population).

    Args:
        survival_rate (:obj:`float`): Defines which fraction of data points
            from previous iterations should be kept for the next iteration.
            Default is 0.2.
        cut_to_iteration_size (:obj:`bool`): defining if, after selecting data
            points, the selection should be cut down to have its length match
            iteration size. Default is :obj:`False` .

    Returns:
        A width controller taking three arguments: the function values `y` ,
        an array of booleans indicating if the corresponding `y` s are sampled
        in the most recent iteration and the iteration_size. See its docstring
        for more information.

    Raises:
        ValueError: The survival_rate should be a number between 0 and 1.
            Provided was (?)."""
    # Check if survival_rate is a number between 0 and 1. If not, raise a
    # ValueError
    if not 0. <= survival_rate <= 1.:
        raise ValueError("The survival_rate should be a number between 0 and "
                         "1. Provided was {}.".format(survival_rate))

    # Define controller
    def controller(y, is_new, iteration_size):
        """ Controller function determining which datapoints should be removed
        at the end of the current iteration.

        This kill controller selects all points that are (1) not sampled in the
        latest iteration and (2) don't form the best `survival_rate` fraction
        of datapoints from the previous iterations. "best" is here defined as
        the `y` values that are the lowest. If the `cut_to_iteration_size`
        parameter of the currying function was set to `True`, the selection
        that this gives is cut down such that the number of selected datapoints
        matches the provided `iteration_size`. This cut is made such that only
        the datapoints with the best `y` values survive.

        This method returns a numpy array of boolean values, in which `True`
        indicates that the data point with matching index should be removed
        at the end of the iteration.

        Args:
            y: `numpy.ndarray` containing the function values of the points
                for which the kill list has to be constructed.
            is_new: `numpy.ndarray` of `bool`s with the same length as `y`. The
                booleans should indicate if the data point with that index
                was sampled in the most recent iteration.
            iteration_size: `int` defining the size to which the selection has
                to be cut down to if `cut_to_iteration_size` of the currying
                function was set to `True`.

        Returns:
            A `numpy.ndarray` with a length equal to the length of `y`, filled
            with booleans. `True` indicates that the corresponding data point
            should be removed. """
        y = y.flatten()
        # The survival rate only applies to data from the previous generations,
        # so we need to determine which data points are not new and find the
        # value for y above which these old points are cut away
        is_old = np.invert(is_new)
        y_sorted = np.sort(y[is_old])
        idx_cut = round(len(y_sorted) * survival_rate) - 1
        cut_a = y_sorted[idx_cut]
        # Now that we have this cut value, we can determine which points to
        # keep through a logical or of is_new (for now we want to keep all new
        # we just sampled) and y < cut.
        to_keep = np.logical_or(is_new, y <= cut_a)
        # If cut_to_iteration_size is set to True, we define a new cut such
        # that exactly `iteration_size` points are kept. We then apply a
        # logical_and over the previous to_keep, to enforce this new cut
        if cut_to_iteration_size:
            y_kept = np.sort(y[to_keep])
            cut_b = y_kept[iteration_size - 1]
            to_keep = np.logical_and(to_keep, y <= cut_b)
        # As this is the *kill* controller, we invert the to_keep array before
        # we return it.
        return np.invert(to_keep)

    return controller


def get_stdev_controller(min_stdev=0.0,
                         max_stdev=np.inf,
                         scales_with_boundary=True,
                         logarithmic=False,
                         inf_replace=1e12):
    """ Get a simple standard deviation controller. See docstring of the
    controller for more information.

    Args:
        min_stdev: Minimum value standard deviations are allowed to have. All
            values below this value will be clipped to `min_stdev`. Default
            value is `0.0`.
        max_stdev: Maximum value standard deviations are allowed to have. All
            values above this value will be clipped to `max_stdev`. Default
            value is `numpy.inf`.
        scales_with_boundary: If set to `True`, the standard deviation scales
            linearly with the ranges allowed by the boundaries provided to
            the controller. Although this is the default, it might not be
            what you want for problem without boundaries. Setting this
            argument to `False` removes the dependence on the boundary
            entirely. See controller docstring for more information.
        logarithmic: `bool` indicating if the controller should run in
            `logarithmic` mode. See controller docstring for more information.
            Default is `False`.
        inf_replace: Number with which infinities in parameter range boundaries
            are replaced. See docstring of the controller for more information.
            Default value is `1e12`.

    Returns:
        Standard deviation controller function. See docstring of the controller
        for more information.

    Raises:
        ValueError: `min_stdev` should be lower than `max_stdev`. Provided were
            (?) and (?) respectively."""
    # Check if mininum stdev is smaller than the maximum stdev.
    if min_stdev > max_stdev:
        raise ValueError("`min_stdev` should be lower than `max_stdev`. "
                         "Provided were {} and {} respectively".format(
                             min_stdev, max_stdev))

    # Define controller
    def controller(width, boundaries, x):
        """ Standard deviation controller that calculates the standard
        deviations of points `x`. The standard deviations are set to `width`
        and are multiplied by the ranges for each dimension, as set by
        `boundaries`, if `scales_with_boundary` in the currying function was
        `True`, and multiplied by `x` if `logarithmic` argument of the currying
        function was set to `True`.

        Post-processing of the calculated standard deviations amounts to
        clipping the values to the `min_stdev` and `max_stdev` defined by the
        currying function and by taking the absolute value of the results.

        Args:
            width: `float` defining the current width parameter.
            boundaries: `None` or `numpy.ndarray` defining the upper and
                lower boundaries of each of the dimensions. If some of the
                boundaries were set to `numpy.inf` (or `-numpy.inf`), these
                values are replaces by `inf_replace` as defined by the
                currying function. `None` is interpreted as 'No boundaries in
                any of the dimensions', resulting in ranges from `-inf_replace`
                to `inf_replace` in all dimensions.
            x: `numpy.ndarray` containing the locations of the data points for
                which the standard deviations need to be calculated. This
                information will only be used if `logarithmic` in the
                currying function was set to `True`.

        Returns:
            `numpy.ndarray` with a length equal to that of `x`, containing the
            calculated standard deviations.

        Raises:
            ValueError: Boundaries of the parameter range have to be provided
                as `None` or as a `numpy.ndarray`. Provided was a(n) (?).
            ValueError: Boundaries provided as an array needs to have a shape
                of `(nDatapoints, nDimensions)` (= (?)). Provided was an array
                of shape (?). """
        # Set standard deviations to its base value: the provided width
        stdevs = width*np.ones(x.shape)
        # Check if need to multiply with ranges as determined in the
        # currying function.
        if scales_with_boundary:
            # Check and convert boundaries to right shape and type
            # If boundaries is None, we replace it with an array filled with
            # -np.inf and np.inf (to be later replaced with the value of
            # `inf_replace`)
            if boundaries is None:
                boundaries = np.ones((x.shape[1], 2)) * np.inf
                boundaries[:, 0] *= -1
            # Else boundaries has to have shape `(x.shape[0], 2)`
            if not isinstance(boundaries, np.ndarray):
                raise ValueError("Boundaries of the parameter range have to "
                                 "be provided as `None` or as a "
                                 "`numpy.ndarray`. Provided was "
                                 "a(n) {}.".format(type(boundaries)))
            if boundaries.shape != (x.shape[1], 2):
                raise ValueError("Boundaries provided as an array needs to "
                                 "have a shape of `(nDatapoints, nDimensions)`"
                                 " (= {}). Provided was an array of "
                                 "shape {}.".format(x.shape, boundaries.shape))
            # Replace infinite values with inf_replace
            boundaries[boundaries == np.inf] = inf_replace
            boundaries[boundaries == -np.inf] = -inf_replace
            # First we determine the ranges of the function, i.e. the size of
            # theparameter space as defined by the boundaries.
            ranges = boundaries[..., 1] - boundaries[..., 0]
            # Linear standard deviations are then given by the width parameter
            # multiplied by this range.
            stdevs = stdevs * ranges
        # If running in logarithmic mode, the standard deviations multiply
        # the result of this calculation by the coordinate. Else, multiply the
        # result by an array of ones to match the right shape
        if logarithmic:
            stdevs = stdevs * x
        # Before returning the stdevs we bound the stdevs using the `min_stdev`
        # and `max_stdev` argument of the decorator function, after which we
        # return the calculated values.
        stdevs[stdevs > max_stdev] = max_stdev
        stdevs[stdevs < min_stdev] = min_stdev
        return np.abs(stdevs)

    return controller
