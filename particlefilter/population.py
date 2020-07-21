# -*- coding: utf-8 -*-
import os
import logging
import numpy as np

from . import utils

# TODO: find out how the logger works
# TODO: implement logger in the code below if useful and easy to implement
logger = logging.getLogger(__name__)


class Population:
    """ `Population` is a container class that stores the data and related
    information for the current iteration. It is 'alive' in the sense that
    the sampling of new data adds new data points to the population (and is
    hence called 'procreate' in the context of this class) and the removal
    of points actually removes them from the container (this is called
    'killing points' in the context of this class).

    The functionality to have data points procreate or to kill data points is
    implemented in this `Population` class, but what is not implemented is the
    functionality to determine the parameters with which this happens. For
    example: to sample new points, standard deviations are needed to form
    normal distributions (to sample from). This functionality is not
    implemented here, but the information from these functionalities is guided
    to instances of this `Population` class via for example the `set_stdevs`
    method. Although `Population` implements a lot of 'smart container'
    methods, it thus needs a conductor of sorts to be fully functional and to
    be actually used in an iterative fashion (as is the intention in the
    context of a particle filter). Within this package this conductor is the
    `ParticleFilter` class.

    Args:
        x: `numpy.ndarray` containing the coordinates of the samples that need
            to form the first generation of data.
        y: `numpy.ndarray` containing the function values for the coordinates
            provided as `x`. Each entry should have just one function value.

    Attributes:
        x: `numpy.ndarray` containing the coordinates of the points currently
            in the population.
        y: `numpy.ndarray` containing the function values associated with the
            coordinates stored in `x`.
        origin_iteration: `numpy.ndarray` containing the iteration in which
            the data points in `x` were added to the population.
        procreation_rates: `numpy.ndarray` containing integer values that
            indicate how many samples will be taken from the samples in `x`.
        stdevs: `numpy.ndarray` containing the standard deviations with which
            the normal distributions will be constructed from which new samples
            will be taken.
        kill_list: `numpy.ndarray` or `list` filled with boolean values. These
            indicate which points will be removed from the population when
            `kill_data` is called. """
    def __init__(self, x, y):
        # Set the base state of the population
        self.reset()
        # Add the first samples `x` and their function values `y` to the
        # population as seed
        self.set_seed(x, y)

    def __len__(self):
        """ Returns the number of points currently in the Population """
        return self.x.shape[0]

    def __bool__(self):
        """ Allows the class to be boolean-ified. `False` will be returned if
        the population object contains no data points. In any other case `True`
        will be returned. """
        return bool(len(self))

    def __str__(self):
        """ Returns a summary of the Population in the form of a string.

        Returns:
            String containing abstracted information about the
            Population. """
        return "Population({} datapoints, {} iteration(s))".format(
            len(self), self._now)

    """ Functions for setting default values """

    def reset(self):
        """ Resets the `Population` and sets placeholder values for all
        its internal variables and arrays. """
        # Iteration counter
        self._now = 0
        # Data point coordinates, function values and iteration in which they
        # were added
        self.x = np.array([])
        self.y = np.array([])
        self.origin_iteration = np.array([])
        # Allocate meta arrays used in processing the data points
        self._reset_procreation_arrays()
        self._reset_kill_arrays()
        # Graveyard file
        self._graveyard_handle = None

    def _reset_procreation_arrays(self):
        """ Allocates the meta arrays containing flags and additional
        information for procreation for each of the stored datapoints. These
        arrays have a length equal to the number of stored datapoints and
        contain only `numpy.nan` entries."""
        self.procreation_rates = np.ones(len(self)) * np.nan
        self.stdevs = np.ones(self.x.shape) * np.nan

    def _reset_kill_arrays(self):
        """ Allocates the meta arrays containing flags and additional
        information using in killing datapoints. These arrays have a length
        equal to the number of stored datapoints and contain only `numpy.nan`
        entries. If `reset_shape_only` is set to
        `False` the content of the arrays is not discarded, but instead the
        array is reinitialised """
        self.kill_list = np.array([False] * len(self))

    def _extend_meta_arrays(self):
        """ This method extends to shape of meta arrays to match the definition
        of the current `x`. This is useful explicitly just after `append`, when
        the meta arrays do not match the shape of `x` anymore. """
        # Get current values
        stdevs = self.stdevs
        rates = self.procreation_rates
        kill_list = self.kill_list
        # Reset meta arrays
        self._reset_procreation_arrays()
        self._reset_kill_arrays()
        # Overwrite entires in meta arrays to old versions
        self.stdevs[:len(stdevs)] = stdevs
        self.procreation_rates[:len(rates)] = rates
        self.kill_list[:len(kill_list)] = kill_list

    """ Smart selection getters """
    # Methods in this section allow the user to quickly get subsets of or
    # metrics about the data stored in this `Population` object.

    def get_data(self):
        """ Returns all stored data point coorindates with their associated
        function values.

        Returns:
            x: numpy.ndarray containing all stored data point coordinates in
                the population. This includes also all coorindates that might
                have been flagged for removal at the end of this iteration.
            y: numpy.ndarray containing the function values for the returned
                `x`. """
        return (self.x, self.y)

    def get_data_with_origin_information(self):
        """ Returns all data point coordinates `x`, their function values
        `y` and their iteration of origin. All these are arrays with the same
        length.

        Returns:
            x: `numpy.ndarray` containing all stored data point coordinates in
                the population. This includes also all coorindates that might
                have been flagged for removal at the end of this iteration.
            y: `numpy.ndarray` containing the function values for the returned
                `x`.
            origin: `numpy.ndarray` containing the iteration ID in which each
                of the data points in `x`was sampled. """
        return (self.x, self.y, self.origin_iteration)

    def get_procreating_data(self):
        """ Returns the coordinates and function values for which the
        procreation rate stored in `procreation_rates` is 1 or larger.

        Returns:
            x: numpy.ndarray containing the stored data point coordinates in
                the population slated for procreation in the current iteration.
                This includes also all coorindates that might have been flagged
                for removal at the end of this iteration.
            y: numpy.ndarray containing the function values for the returned
                `x`. """
        return (self.x[self.procreation_rates > 0],
                self.y[self.procreation_rates > 0])

    def get_data_on_kill_list(self):
        """ Returns the coordinates and function values for which the removal
        flag in the `kill_list` property is `True`.

        Returns:
            x: numpy.ndarray containing the stored data point coordinates in
                the population slated for removal at the end of the current
                iteration.
            y: numpy.ndarray containing the function values for the returned
                `x`. """
        return (self.x[self.kill_list], self.y[self.kill_list])
    
    def get_data_on_kill_list_with_origin(self):
        """ Returns the coordinates, function values and origin iteration for
        which the removal flag in the `kill_list` property is `True`.

        Returns:
            x: numpy.ndarray containing the stored data point coordinates in
                the population slated for removal at the end of the current
                iteration.
            y: numpy.ndarray containing the function values for the returned
                `x`.
            origin: `numpy.ndarray` containing the iteration ID in which each
                of the data points in `x`. """
        return (self.x[self.kill_list],
                self.y[self.kill_list],
                self.origin_iteration[self.kill_list])

    def get_latest_data(self):
        """ Returns all data point coordinates `x` and associated function
        values `y` that were added in the most recent addition operation.

        Returns:
            x: numpy.ndarray containing the data point coordinates that were
                most recently added.
            y: numpy.ndarray containing the function values for the returned
                `x`. """
        latest_id = np.amax(self.origin_iteration)
        return self.get_data_by_origin(latest_id)

    def get_data_by_origin(self, origin_id):
        """ Returns all data point coorindates `x` and associated function
        values `y` that originated from the provided origin

        Args:
            origin_id: `int` indicating the origin iteration of the data
                to be returned.

        Returns:
            x: numpy.ndarray containing the data point coordinates that were
                added in the indicated iteration.
            y: numpy.ndarray containing the function values for the returned
                `x`. """
        origin_id = int(origin_id)
        indices = self.origin_iteration == origin_id
        return (self.x[indices], self.y[indices])

    def get_function_extremes(self):
        """ Returns the extreme values of the function values `y` currently
        stored in the `Population` object.

        Returns:
            (`min`, `max`): Minimum and maximum of the stored function values.
        """
        return (np.amin(self.y), np.amax(self.y))

    def get_means_and_stdevs_for_sampling(self):
        """ This method returns the means and the standard deviations of the
        data points needed for sampling. As such, this method requires the
        procreation rate and the standard deviations to be set.

        Returns:
            `means` and `stdevs` of the normal distributions to sample. If
            the procreation rate of some of the stored data points `x` was
            larger than 2, the mean and standard deviation of that data point
            occur multiple times in these arrays.

        Raises:
            Exception: To get the mean and standard deviation of the current
                population of stored data points for sampling, the procreation
                rate has to be set first. You can do this through the
                `set_procreation_rates` method.
            Exception: To get the mean and standard deviation of the current
                population of stored data points for sampling, the standard
                deviations have to be set first. You can do this through the
                `set_stdevs` method."""
        # Check if procreation rate and stdev are not an arrays of nans
        if np.sum(np.isnan(self.procreation_rates)) == np.prod(self.x.shape):
            raise Exception("To get the mean and standard deviation of the "
                            "current population of stored data points for "
                            "sampling, the procreation rate has to be set "
                            "first. You can do this through the "
                            "`set_procreation_rates` method.")
        if np.sum(np.isnan(self.stdevs)) == np.prod(self.x.shape):
            raise Exception("To get the mean and standard deviation of the "
                            "current population of stored data points for "
                            "sampling, the standard deviations have to be set "
                            "first. You can do this through the `set_stdevs` "
                            "method.")
        # Construct indices from procreation rates and select data with these
        # indices
        indices = self._get_indices_from_rates()
        means = self.x[indices]
        stdevs = self.stdevs[indices]
        return means, stdevs

    """ Methods for validation and reshaping of input """
    # The `Population` class expects input given by the user to be of a
    # certain type and shape. The methods defined in this section apply checks
    # to a wide range of input given by the user and return nicely formatted
    # error messages if things are not as expected. These methods also reshape
    # the data (if possible) to the shape required by the algorithm. All
    # methods in this section return the (reshaped) input they were supplied
    # with (except for the `self` argument).

    def validate_input_data(self, x, y):
        """ Checks if the input arrays `x` and `y` are `numpy.ndarray`s and
        have the correct shape.

        The coorindates of the input data `x` should be at least 2-dimensional
        and the first axis should represent the data points. The function
        values for these coordinates, given in `y`, should therefore have the
        same first dimension length. This values array should additionally be
        exactly 1- or 2-dimensional (the second dimension should have length
        1).

        Args:
            x: numpy.ndarray containing the coordinates of samples.
            y: numpy.ndarray containing the function values for the coordinates
                provided as `x`.

        Returns:
            Input arrays `x` and `y`. The `y` array is reshaped to a
            2-dimensional array if a 1-dimensional array was provided.

        Raises:
            TypeError: Data points in `x` should be provided in a
                `numpy.ndarray`.
            TypeError: Function values for provided datapoints should
                be in the form of a `numpy.ndarray`.
            ValueError: Data points `x` should be at least 2-dimensional, with
                the first axis representing the datapoints.
            ValueError: Function values for data points should be provided as a
                1- or 2-dimensional array, the first axis representing the
                datapoints. If a 2-dimensional array is provided, the second
                axis should have length 1.
            ValueError: The number of data point coordinates in `x` does not
                match the number of provided function values in `y`. """
        # Check if `x` and `y` are arrays
        if not isinstance(x, np.ndarray):
            raise TypeError("Data points in `x` should be provided in a "
                            "`numpy.ndarray`.")
        if not isinstance(y, np.ndarray):
            raise TypeError("Function values for provided datapoints should "
                            "be in the form of a `numpy.ndarray`.")
        # Check if shape of `x` is at least 2-dimensional
        if len(x.shape) < 2:
            raise ValueError("Data points `x` should be at least "
                             "2-dimensional, with the first axis representing "
                             "the datapoints.")
        n = len(x)
        # Check shape of `y`
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y.shape) != 2 or y.shape[1] != 1:
            raise ValueError("Function values for data points should be "
                             "provided as a 1- or 2-dimensional array, the "
                             "first axis representing the datapoints. If a "
                             "2-dimensional array is provided, the second "
                             "axis should have length 1.")
        if y.shape[0] != n:
            raise ValueError("The number of data point coordinates in `x` "
                             "does not match the number of provided function "
                             "values in `y`.")
        # Return arrays, so that possible reshaping can be used
        return (x, y)

    def validate_procreation_rates(self, rate):
        """ Validates the length of an array of procreation rates and converts
        the entries to int32.

        The procreation rate should be a 1-dimensional array of length equal to
        the number of datapoints in the population. When stored in the object,
        its values will be converted to `int32`. An error is raised if this
        conversion changed the values in the array.

        Args:
            rate: numpy.ndarray of integers with length equal to the number of
                entries in the Population.

        Returns:
            The provided procreation rates with its entries casted to `int32`.

        Raises:
            TypeError: The provided procreation rate per data point should be
                a `numpy.ndarray`. Provided was a(n) (?).
            ValueError: The provided procreation rate per data point should be
                a `numpy.ndarray` with a length equal to the number of points
                currently in the population (?).
            ValueError: The procreation rates could not be converted to int32
                without loss of information. This can result in unexpected and
                unwanted behaviour when sampling the next generation of
                datapoints. """
        # Check if rate is a numpy array of length equal to the number of
        # points currently in the population
        if not isinstance(rate, np.ndarray):
            raise TypeError("The provided procreation rate per data point "
                            "should be a `numpy.ndarray`. Provided was a(n) "
                            "{}.".format(type(rate)))
        if len(rate) != len(self):
            raise ValueError("The provided procreation rate per data point "
                             "should be a `numpy.ndarray` with a length equal "
                             "to the number of points currently in the "
                             "population {}. Provided array had a length "
                             "of {}".format(len(self), len(rate)))
        # Convert rates to int32 and raise a warning if this changes the values
        # in the array (which might yield unwanted behaviour at procreation).
        rate_int = rate.astype(np.int32)
        if np.sum(np.abs(rate - rate_int)) != 0:
            raise ValueError("The procreation rates could not be converted to "
                             "int32 without loss of information. This can "
                             "result in unexpected and unwanted behaviour "
                             "when sampling the next generation of "
                             "datapoints.")
        return rate_int

    def validate_stdevs(self, stdevs, procreating_only=True):
        """ Validates provided standard deviations used in sampling the next
        generation of data points.

        Standard deviations should be provided as a `numpy.ndarray` with a
        shape equal to either the shape of the currently stored data point
        coordinates `x` or have a shape that is reshapable to `(n, )`, where
        `n` is the number of currently stored datapoints. In this last case
        this method will reshape the array to match the shape of the currently
        stored coordinates `x`.

        The `procreating_only` flag indicates if the provided standard
        deviations are for all stored coordinates `x` (`False`) or if they
        belong to the procreating points only (`True`, default). If `True`, the
        shape of (or number of elements in) `stdevs` should match the shape
        (or length) of the array containing only the procreating points. This
        array can be accessed through the `get_procreating_data` method.

        Args:
            stdevs: `numpy.ndarray` containing the standard deviations used to
                construction normal distributions used for sampling the next
                generation.
            procreating_only: indicates if the provided standard
                deviations are for all stored coordinates `x` (`False`) or if
                they belong to the procreating points only (`True`, default).

        Returns:
            The provided standard deviations `stdevs`, reshaped if necessary
            (see docstring above for information on when this happens).

        Raises:
            TypeError: When setting `procreating_only` to `True`, you should
                provide an `stdevs` array with as many entries in the first
                axis as there are points that will be procreating. You provided
                (?) entries, but (?) were expected.
            ValueError: Standard deviations should have a shape equal to the
                coordinates `x` stored in the population or be castable to the
                shape `(n, 1)`, with `n` the number of points in `x`. Provided
                was (?)."""
        # Check the type of the provided standard deviations
        if not isinstance(stdevs, np.ndarray):
            raise TypeError("Standard deviations should be provided as a "
                            "`numpy.ndarray`. Provided was "
                            "a(n) {}.".format(type(stdevs)))
        # If `procreating_only` was set to `True`, we want to convert it first
        # to a full array for all data (makes the rest of the code easier
        # to understand).
        if procreating_only:
            full_stdevs = np.empty(self.x.shape)
            idx = (self.procreation_rates > 0)
            # Check if the shape of the provided stdevs matches the expected
            # shape
            if full_stdevs[idx].shape[0] != stdevs.shape[0]:
                raise ValueError("When setting `procreating_only` to `True`, "
                                 "you should provide an `stdevs` array with "
                                 "as many entries in the first axis as there "
                                 "are points that will be procreating. You "
                                 "provided {} entries, but {} were "
                                 "expected.".format(stdevs.shape[0],
                                                    full_stdevs[idx].shape[0]))
            if full_stdevs[idx].shape == stdevs.shape:
                full_stdevs[idx] = stdevs
            if len(stdevs.shape) == 1 or (len(stdevs.shape) == 2
                                          and stdevs.shape[1] == 1):
                for i in range(self.x.shape[1]):
                    full_stdevs[idx, i] = stdevs
            stdevs = full_stdevs
        # Standard deviations are provided for *all* data in the
        # population. Let's check if the shapes match. If they do,
        # everything is okay and we can immediately return the provided
        # stdevs.
        if stdevs.shape == self.x.shape:
            return stdevs
        # The shapes did not match, so let's check if what was provided was
        # the stdevs for each datapoint individually. If not, raise a
        # ValueError.
        if np.prod(stdevs.shape) != len(self):
            raise ValueError("Standard deviations should have a shape "
                             "equal to the coordinates `x` stored in the "
                             "population or be castable to the shape "
                             "`(n, 1)`, with `n` the number of points in "
                             "`x`. Provided was {}, expected "
                             "was {} or ({}, ).".format(
                                 stdevs.shape, self.x.shape, self.x.shape[0]))
        # Provided stdevs were indeed 1-per-point provided. We expand this
        # and return the result
        stdevs = stdevs.reshape(-1, 1)
        return stdevs * np.ones(self.x.shape)

    def validate_kill_list(self, kill_list):
        """ Validates the provided `kill_list` for its length and contents.

        `kill_list`s should have a length equal to the number of data points
        currently contained in the Population object and should only
        contain boolean values or values that could be converted to booleans.

        Although the name `kill_list` seems to suggest that it should be a
        list, it is also allowed to be a `numpy.ndarray`. If it is indeed a
        numpy array, not the length but the number of entries in the array
        should match the number of data points in the population.

        Args:
            kill_list: `list` or `numpy.ndarray` containing boolean values.

        Returns:
            The provided kill_list, reshaped to `(n, 1)` if provided as a
            `numpy.ndarray`, where `n` equals the number of data points
            currently stored in the population.

        Raises:
            TypeError: `kill_list`s should be `numpy.ndarray`s or `list`s.
                Provided was a(n) (?).
            ValueError: The length of the provided `kill_list` (?) does not
                match the number of data points currently in the
                `Population` object (?)."""
        # Validate that the kill_list is a numpy array or a list
        if not isinstance(kill_list, (list, np.ndarray)):
            raise TypeError("`kill_list`s should be `numpy.ndarray`s or "
                            "`list`s. Provided was a(n) {}.".format(
                                type(kill_list)))
        if isinstance(kill_list, np.ndarray):
            kill_list = kill_list.reshape(-1, )
        # Check the length of the provided kill list
        if len(kill_list) != len(self):
            raise ValueError("The length of the provided `kill_list` {} does "
                             "not match the number of data points currently "
                             "in the `Population` object {}.".format(
                                 len(kill_list), len(self)))
        return kill_list

    """ Data methods """
    # Data methods allow the addition of new data to the Population object
    # and the configuration of meta arrays, like `procreation_rates` and
    # `kill_list`. Although all these values are stored in properties of the
    # object and the user *could* alter these properties directly, these
    # methods apply checks to the input so that errors in the rest of the code
    # are avoided (and nice error messages are given when things are not as
    # expected).

    def set_seed(self, x, y):
        """ Sets the seed (i.e. the initial data) to be contained in this
        Population.

        Provided data is checked for being numpy arrays. The `x` array should
        be at least 2-dimensional, the `y` array *has* to be 2-dimensional.

        If there is already data in the `Population` object, this method
        will raise an Exception.

        Args:
            x: numpy.ndarray containing the coordinates of the samples that
                need to be added to the `Population` object. The shape
                of each entry in this array should match the shape of the
                already stored entries.
            y: numpy.ndarray containing the function values for the coordinates
                provided as `x`. Each entry should have just one function
                value.

        Raises:
            Exception: Seed cannot be set, as there is already data present in
                the population. The population can be emptied by calling the
                `reset()` method.
        """
        # Check if there is already data present in the population. If there
        # is,raise and Exception.
        if self:
            raise Exception("Seed cannot be set, as there is already data "
                            "present in the population. The population can be "
                            "emptied by calling the `reset` method.")
        # Check the shapes of `x` and `y`.
        x, y = self.validate_input_data(x, y)
        # Store provided data and set origin iteration to 0 for these samples
        self.x = x
        self.y = y
        self.origin_iteration = np.zeros(len(x))
        # Allocate meta arrays and increase iteration counter to be 1
        self._reset_procreation_arrays()
        self._reset_kill_arrays()
        self._now = 1

    def append(self, x, y):
        """ Appends samples `x` and their function values `y` to the population
        and store for these samples in which iteration they were added.

        Args:
            x: numpy.ndarray containing the coordinates of the samples that
                need to be added to the `Population` object. The shape
                of each entry in this array should match the shape of the
                already stored entries.
            y: numpy.ndarray containing the function values for the coordinates
                provided as `x`. Each entry should have just one function
                value.

        Raises:
            ValueError: The shape of the points in `x` (?) does not match the
                shape of already stored datapoints (?).
            ValueError: Only scalar function values `y` can be stored in a
                `Population` object. Provided were entries with shape
                (?)."""
        # Validate the inherent required type and shape of `x` and `y`
        x, y = self.validate_input_data(x, y)
        # Check if `x` is an array and shape is correct
        if x.shape[1:] != self.x.shape[1:]:
            raise ValueError("The shape of the points in `x` {} does not "
                             "match the shape of already stored "
                             "datapoints {}".format(x.shape[1:],
                                                    self.x.shape[1:]))
        # Add x and y to population
        self.x = np.vstack((self.x, x))
        self.y = np.vstack((self.y, y))
        # Extend self.origin_iteration
        iteration_array = np.ones(len(x)) * self._now
        self.origin_iteration = np.hstack(
            (self.origin_iteration, iteration_array))
        # Allocate meta arrays and fill with the old data so that it is not
        # lost
        self._extend_meta_arrays()

    def set_procreation_rates(self, rates):
        """ Sets for each of the datapoints the procreation rate.

        The procreation rate should be a 1-dimensional array of length equal to
        the number of datapoints in the population. When stored in the object,
        its values will be converted to `int32`. The sum of these values is the
        number of data points that will be sampled in the next iteration.

        Args:
            rate: numpy.ndarray of integers with length equal to the number of
                entries in the Population. """
        # Validate the provided `rate` array for correctness and convert to
        # int32.
        rates = self.validate_procreation_rates(rates)
        # Store procreation rate as an int32 array.
        self.procreation_rates = rates.flatten()

    def set_stdevs(self, stdevs, procreating_only=True):
        """ Sets the standard deviations for the normal distributions that
        will be sampled for the next generation of data points.

        Standard deviations should be provided as a `numpy.ndarray` with a
        shape equal to either the shape of the currently stored data point
        coordinates `x` or have a shape that is reshapable to `(n, )`, where
        `n` is the number of currently stored datapoints. In this last case
        this method will reshape the array to match the shape of the currently
        stored coordinates `x`. The values of this array then match the
        standard deviation in each of the dimensions of the function.

        The `procreating_only` flag indicates if the provided standard
        deviations are for all stored coordinates `x` (`False`) or if they
        belong to the procreating points only (`True`, default). If `True`, the
        shape of (or number of elements in) `stdevs` should match the shape
        (or length) of the array containing only the procreating points. This
        array can be accessed through the `get_procreating_data` method.

        Args:
            stdevs: `numpy.ndarray` containing the standard deviations used to
                construction normal distributions used for sampling the next
                generation.
            procreating_only: indicates if the provided standard
                deviations are for all stored coordinates `x` (`False`) or if
                they belong to the procreating points only (`True`, default).
        """
        # Validate the type and shape of the provided standard deviations
        stdevs = self.validate_stdevs(stdevs, procreating_only)
        self.stdevs = stdevs

    def set_kill_list(self, kill_list):
        """ Sets the `kill_list` property, which defines through boolean values
        which data points need to be removed at the end of the current
        iteration.

        Args:
            kill_list: `list` or `numpy.ndarray` containing boolean values. """
        # Validate type (and shape, if numpy.ndarray) of the provided kill_list
        kill_list = self.validate_kill_list(kill_list)
        self.kill_list = kill_list

    """ Iteration methods """
    # Iteration methods are used to control the iteration. Examples of
    # iteration methods are the `procreate` method, used to sample new data
    # points from provided gaussian stdevs and procreation rates, and the
    # `kill_data` method, that removes all data point coordinates and function
    # values from the population that were slated for removal in the
    # `kill_list` array.

    def procreate(self, boundaries=None, max_attempts=100):
        """ Creates samples for the next generation based on the
        `procreation_rates` and `stdevs` provided earlier by the user through
        their respective setters.

        The number of samples that will be generated, and from which normal
        distributions these samples will be taken, is determined by the
        `procreation_rates`. Data coorindates `x` with a rate of `r` will
        get `r` samples taken out of their normal distribution. The width
        of this normal distribution is given by `stdevs`.

        As samples are taken using a multivariate normal distribution, there
        is a possibility that some samples fall outside the region of interest.
        The `boundaries` argument defines this region of interest. If points
        fall indeed outside of bounds, these (and only these) points are
        resampled until they do. Setting `boundaries` to `None` is equivalant
        to setting no boundaries at all.

        Through the `max_attempts` argument the maximum number of runs of this
        resampling procedure can be defined. If there are still points that
        fall outside the defined boundaries after the maximum number of
        attempts has been reached, these points are simply not returned.

        At the end of the `procreate` method all information needed for
        procreation (i.e. the procreation rates and stdevs) is deleted, so that
        accidental double execution is not possible.

        Args:
            boundaries: `None` or `numpy.ndarray` defining the boundaries
                of the parameter space within which the samples need to fall.
                See the documentation for the `utils.validate_boundaries`
                method for more information on the formatting of this
                argument.
            max_attempts: integer defining the maximum number of resampling
                attempts for points that fall outside of the defined region
                of interest.

        Returns:
            `numpy.ndarray` of shape `(nSamples, nDimensions)`"""
        # To avoid overhead, we just perform simple tests on the shape and type
        # of boundaries. If these go wrong, we run the full
        # `utils.validate_boundaries` checks
        run_validation = False
        if isinstance(boundaries, np.ndarray):
            if boundaries.shape != (*self.x.shape[1:], 2):
                run_validation = True
        elif boundaries is not None:
            run_validation = True
        if run_validation:
            boundaries = utils.validate_boundaries(boundaries,
                                                   self.x.shape[1:])
        # Get an array with the datapoints in `x` with a frequency defined in
        # the `procreation_rates` property. Apply this array to get the means
        # and stdevs of the normal distributions from which we will sample.
        means, stdevs = self.get_means_and_stdevs_for_sampling()
        # Create covariance matrices
        covs = self._calculate_covariance_matrices(stdevs)
        # Perform sampling until all points are within boundary (and select
        # only points that lie within this boundary) or until the maximum
        # number of attempts has been reached
        out_of_bounds = np.array([True] * means.shape[0])
        samples = np.zeros(means.shape)
        attempts = 0
        while np.sum(out_of_bounds) != 0 and attempts <= max_attempts:
            samples[out_of_bounds] = self._sample(means[out_of_bounds],
                                                  covs[out_of_bounds])
            out_of_bounds[out_of_bounds] = self._is_out_of_bounds(
                samples[out_of_bounds], boundaries)
            attempts += 1
        # If maximum number of attempts has been reached, filter the samples
        # such that only the ones within bounds are returned
        if attempts >= max_attempts:
            samples = samples[np.invert(out_of_bounds)]
        # Samples have been succesfully taken, so let's return them.
        return samples

    def _calculate_covariance_matrices(self, stdevs):
        """ Constructs diagonal covariance matrices from the provided standard
        deviations.

        Args:
            stdevs: Standard deviations in the form of a `numpy.ndarray` of
                shape `(nDatapoints, nDimensions)`.

        Returns:
            An array of shape `(nDatapoints, nDimensions, nDimensions)`
            representing the covariance matrices belonging to the provided
            standard deviations. """
        covs = np.zeros((stdevs.shape[0], stdevs.shape[1], stdevs.shape[1]))
        for i in range(stdevs.shape[1]):
            covs[:, i, i] = stdevs[:, i]
        return np.power(covs, 2)

    def _get_indices_from_rates(self):
        """ Converts the `procreation_rates` array to an array of indices that
        need to be selected. This conversion takes into account the rate as
        defined in the `procreation_rates` array, such that a rate of 3 results
        in the corresponding index appearing three times in the returned array.

        Returns:
            Array of the indices for the data points in `x` and `stdevs` that
            need to be taken for the sampling of the next generation. These
            indices appear with the correct frequency in this returned
            array."""
        indices = []
        for i, entry in enumerate(self.procreation_rates):
            indices.extend([i] * entry)
        indices = np.array(indices)
        return indices

    def _sample(self, means, covs):
        """ Sample points from multivariate normal distributions.

        Args:
            mean: `numpy.ndarray` containing the means of the normal
                distributions in all dimensions from which samples have to be
                taken.
            covs: `numpy.ndarray` containing square covariance matrices for the
                normal distributions.

        Returns:
            Samples taken from the normal distributions defined throught the
            input. """
        # Allocate space for the samples we have to take
        samples = np.zeros(means.shape)
        # Loop over all samples to take and take them
        for i in range(means.shape[0]):
            samples[i] = np.random.multivariate_normal(means[i], covs[i])
        return samples

    def _is_out_of_bounds(self, samples, boundaries):
        """ Checks if the provided samples are outside the defined boundaries.
        Returns a list of booleans, being `True` if the specific sample lies
        outside the boundaries and `False` otherwise.

        Args:
            samples: `numpy.ndarray` defining the samples that have to be
                checked.
            boundaries: `None` or `numpy.ndarray`. If `None`, this indicates
                that there are no boundaries to take into account and a list
                full of `False` will be returned. If it is a `numpy.ndarray`,
                this array should have been validated by the
                `utils.validate_boundaries` method.

        Returns:
            A list of length equal to the number of samples provided, filled
            with boolean values. The indices of the list correspond to the
            indices of the samples in `samples`. A `False` indicates that the
            point was within the defined boundaries, `True` indicates that the
            point was outside of these bounds. """
        # Check if boundaries is `None`. This is the easiest case, as this
        # indicates the absence of any boundaries. In this case we can just
        # return a list full of `False`.
        if boundaries is None:
            return [False] * len(samples)
        # If we have come here, this means we have to do more work and actually
        # check the boundaries on all samples. We do this first for the lower
        # boundary and then for the upper boundary.
        below_lower = samples - boundaries[..., 0].reshape(-1) >= 0
        above_upper = samples - boundaries[..., 1].reshape(-1) <= 0
        # We combine these arrays with an or operation and return this array
        element_level = np.logical_and(below_lower, above_upper)
        sample_level = np.prod(element_level, axis=1) == 0
        return sample_level

    def kill_data(self):
        """ Removes all data from the population that was slated for removal
        through the `kill_list` boolean iterable.

        This operation cannot be undone, but data can be kept by using the
        `save` method before executing this method. The stored `kill_list` is
        reset after this operation to prevent accidental double execution (pun
        definitely intended)."""
        # Send data to graveyard
        if self.has_graveyard():
            self.send_to_graveyard(*self.get_data_on_kill_list_with_origin())
        # Get indices of the entries to KEEP
        indices = np.invert(self.kill_list)
        # Apply selection
        self.x = self.x[indices]
        self.y = self.y[indices]
        self.origin_iteration = self.origin_iteration[indices]
        # Reset the kill arrays to prevent double executiong
        self._reset_procreation_arrays()
        self._reset_kill_arrays()

    def end_iteration(self):
        """ Wraps up the current iteration by increasing the iteration counter
        and killing unkilled data (if applicable)."""
        # Kill data
        self.kill_data()
        # Increase iteration counter
        self._now += 1

    """ Storage and abstraction methods """
    # These storage and abstraction methods allow the user to store the current
    # state of the Population object and to get general information about
    # it.

    def has_graveyard(self):
        """ Checks if a graveyard is defined with the `make_graveyard` method.

        Returns:
            Boolean indicating if the graveyard has been defined (`True`) or
            not (`False`). """
        if hasattr(self, '_graveyard_handle'):
            return self._graveyard_handle is not None
        return False

    def make_graveyard(self, filename=None):
        """ Creates a graveyard file to which, before the end of each
        iteration, the datapoints that are killed are written to, together with
        their function values `y` and iteration of origin. Data is stored in
        .csv format.

        Args:
            filename: Path to which the data should be written. If set to
                `None`, the graveyard will be considered 'not set' and no 
                killed data will be stored. """
        # Close previous graveyard (if exists)
        if self.has_graveyard():
            self._graveyard_handle.close()
        self._graveyard_handle = None
        # Create new graveyard, if asked
        if filename is not None:
            self._graveyard_handle = open(os.path.abspath(filename), 'w')
            self._graveyard_has_header = False
    
    def send_to_graveyard(self, x, y, origin):
        """ Store provided data in the graveyard file, defined with the
        `make_graveyard` method.

        Args:
            x: `numpy.ndarray` containing all stored data point coordinates in
                the population. This includes also all coorindates that might
                have been flagged for removal at the end of this iteration.
            y: `numpy.ndarray` containing the function values for the returned
                `x`.
            origin: `numpy.ndarray` containing the iteration ID in which each
                of the data points in `x`was sampled. """
        if self.has_graveyard():
            # Write header if not already done
            if not self._graveyard_has_header:
                header = ['current_iteration', 'origin_iteration'] + ['x'+str(i) for i in range(len(x[0]))] + ['y']
                self._graveyard_handle.write(','.join(header) + "\n")
                self._graveyard_has_header = True
            # Populate with data
            data = np.hstack((np.ones((len(x), 1))*self._now,
                              x, y.reshape(-1, 1),
                              origin.reshape(-1, 1))).astype(np.str).tolist()
            addition = "\n".join([','.join(data[i]) for i in range(len(data))])
            self._graveyard_handle.write(addition+"\n")
            self._graveyard_handle.flush()

    def save(self, filepath):
        """ This method stores the current content of the Population to a
        file in .csv format. Stored are `x`, `y`, `origin_iteration`,
        `procreation_rates`, `stdevs` and `kill_list`, but only if they are
        present in the Population to begin with.

        filepath: Path to the file that will be created / overwritten to store
            the data in .csv format. """
        # Create master array to store
        headers = ["x{}".format(i) for i in range(self.x.shape[1])]
        headers.extend(["y", "origin_iteration", "procreation_rate"])
        headers.extend(
            ["stdev{}".format(i) for i in range(self.stdevs.shape[1])])
        headers.extend(["on_kill_list"])
        master = np.hstack(
            (self.x, self.y.reshape(-1,
                                    1), self.origin_iteration.reshape(-1, 1),
             self.procreation_rates.reshape(-1, 1), self.stdevs,
             np.array(self.kill_list).reshape(-1, 1)))

        # Store file at provided location
        np.savetxt(filepath,
                   master,
                   delimiter=',',
                   header=','.join(headers),
                   comments='')
