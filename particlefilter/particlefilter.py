# -*- coding: utf-8 -*-
import logging
from inspect import signature
import numpy as np

from .controllers import (get_width_controller, get_stdev_controller,
                          get_kill_controller)
from .population import Population
from . import utils
from . import samplers

# TODO: find out how the logger works
# TODO: implement logger in the code below if useful and easy to implement
logger = logging.getLogger(__name__)


class ParticleFilter:
    """ The `ParticleFilter` class implements the gaussian particle filter
    optimisation algorithm in pure `numpy` and python.

    The core of the algorithm is formed by the `population` property, that
    stores an instance of `population.Population`. The `ParticleFilter` can be
    seen as a conductor of sorts of this object.

    One of the simplest usages of the `ParticleFilter`, that runs with all
    default settings is the following code. It defines data and runs 10
    iterations of the particle filter algorithm on it in just 10 lines (!).

        def function(x):
            return np.sum(np.power(x, 2), axis=1)

        x = np.random.rand(100, 2)
        y = label(x)

        pf = ParticleFilter(function, 100,
                            boundaries=np.array([[0, 1], [0, 1]]))
        pf.set_seed(x, y)
        pf.initialise_run()
        for iteration in range(10):
            pf.run_iteration()

    To make the code more transparent, the `ParticleFilter` class implements
    user-definable callbacks. These callbacks should take four parameters:
    the iteration number, the current width parameter, the function defined
    at construction of the `ParticleFilter` object, and the `Population`
    object (that contains all current data). The `set_callback` method can be
    used to link your own functions to a callback handle. The docstring of the
    `validate_callback_name` method lists all possible callback handles.

    Args:
        function: `callable` taking 1 `numpy.ndarray` containing coordinates
            in the parameter space you want to optimise and returns a
            `numpy.ndarray` containing the function values of those data
            points. See docstring of `validate_function` for more information.
        iteration_size: `int` defining how many new points need to be sampled
            at each iteration of the algorithm.
        initial_width: Initial width parameter (`float`). How this parameter
            is used, depends on the `stdev_controller`. Default is 2.
        boundaries: Definition of the edges of the region of interest, within
            which all samples must lie. See docstring of `validate_boundaries`
            for more information on formatting of the argument. Default is
            `None`.
        width_controller: A callable that takes the iteration number and the
            width of the previous iteration and transforms it into a new width.
            See the `controllers.get_width_controller` function for a
            convenient implementation. Default is
            `controller.get_width_controller()`.
        stdev_controller: A callable that takes the width parameter, sampling
            boundaries and the coordinates `x` of already samples points and
            returns a `numpy.ndarray` of shape `(nDatapoints, nDimensions)`,
            containing standard deviations which are used to form normal
            distributions to sample from.  See
            `controllers.get_stdev_controller` for a convenient implementation.
            Default is `controllers.get_stdev_controller()`.
        kill_controller: A callable that takes function values `y`, a boolean
            array `is_new` indicating for each of the entries in `y` if they
            were sampled in the last iteration, and an integer `iteration_size`
            (see the `iteration_size` argument above). The callable should
            return a list or numpy array containing booleans that indicate
            which points should be removed from the Population. See
            `controllers.get_kill_controller` for a convenient implementation.
            Default is `controllers.get_kill_controller()`.
        max_resample_attempts: Integer indicating the maximum number of
            attempts points will be resampled from their normal distributions
            if they were sampled outside of `boundaries`.
        inf_replace: Value to replace `np.inf` with in `boundaries`, if it
            happens to appear there. `-np.inf` is replaced with `-inf_replace`.

    Attributes:
        boundaries: Boundaries within which all samples must be sampled
        function: Callable that gives for coordinates `x` the function values
            `y`
        inf_replace: Value with with infinities are placed in boundaries if
            they occur.
        initial_width: The initial width parameter at the start of a run. Its
            value will remain the same over a run.
        iteration: The iteration counter. Starts at 0 (for sampling) and is
            incremented with 1 at the start of each iteration (in contrast
            to the `width` parameter).
        iteration_size: Number of points sampled based on the current
            population in each iteration.
        kill_controller: Function controlling which points will die at the
            end of an iteration. See `controllers.get_kill_controller` for
            more informatoin.
        max_resample_attempts: Maximum number of attempts a point that is
            sampled outside of the boundaries defined by `boundaries` will be
            resampled.
        population: The `population.Population` object containing the data,
            related information (like standard deviations for each of the
            points) and functionality to sample new data.
        stdev_controller: Function that calculates the standard deviations of
            the population data. See `controllers.get_stdev_controller` for
            more information.
        width: Current width parameter. This parameter can change over
            iterations (in contrast to the `initial_width` parameter).
        width_controller: Function controlling how the `width` parameter
            changes over iterations. See `controllers.get_width_controller` for
            more information. """
    def __init__(self,
                 function,
                 iteration_size,
                 initial_width=2,
                 boundaries=None,
                 width_controller=get_width_controller(),
                 stdev_controller=get_stdev_controller(),
                 kill_controller=get_kill_controller(),
                 max_resample_attempts=100,
                 inf_replace=1e12):
        self.function = self.validate_function(function)
        self.iteration_size = int(iteration_size)
        self.set_boundaries(boundaries)
        self.initial_width = initial_width
        self.width_controller = width_controller
        self.stdev_controller = stdev_controller
        self.kill_controller = kill_controller
        self.max_resample_attempts = max_resample_attempts
        self.inf_replace = np.abs(inf_replace)

        self._init_callbacks_dictionary()
        self.reset()

    def reset(self):
        """ Resets all properties of the `ParticleFilter` object related to
        current iterations to their default values. These properties are the
        current iteration (which is set to 0), the stored data (which is
        entirely removed, including the seed) and the width (which is reset
        to its initial value). """
        self.iteration = 0
        self.population = None
        self.width = self.initial_width

    """ Validation methods """
    # To make sure there are as few difficult to understand errors, these
    # validation methods explicitly check the user input for their
    # correspondence with assumptions in the input. This allows us to give
    # back nice error messages that are actually understandable by users
    # (at least, hopefully).

    def validate_callback_name(self, name):
        """ Checks if there exists a callback with provided name.

        Currently implemented callbacks are:
        - after_population_init
        - at_start_of_run
        - at_start_of_iteration
        - before_width_controller_call
        - after_width_controller_call
        - before_selection
        - after_selection
        - before_stdev_controller_call
        - after_stdev_controller_call
        - before_procreation
        - after_procreation
        - before_kill_controller_call
        - after_kill_controller_call
        - before_kill_data
        - at_end_of_iteration

        Args:
            name: Name of the callback which has to be checked for existence.

        Returns:
            Name of the handle, the same as provided as input argument.

        Raises:
            ValueError: Callback name '(?)' not recognised. See the docstring
                of the `validate_callback_name` method for a list of
                implemented (and therefore allowed) callback names. """
        # Check if `name` exists for callbacks
        if name not in self._callbacks:
            raise ValueError("Callback name '{}' not recognised. See the "
                             "docstring of the `validate_callback_name` "
                             "method for a list of implemented (and therefore "
                             "allowed) callback names.".format(name))
        return name

    def validate_callback(self, function):
        """ Checks if the provided callback conforms to the exectation.

        Callbacks are allowed to be a callable or `None`. `None` indicates that
        said callback will not be used. If a callable is provided, this
        callable should take four arguments:
        - The iteration number (`int`)
        - The current width (`float`)
        - The function that generates `y` values (`function`)
        - The current data population (`Population`).

        Args:
            function: The callback that you want to have checked for validity.

        Returns:
            The same function as provided in the input.

        Raises:
            TypeError: Callbacks can only be callable objects or `None`.
                Provided was (?).
            ValueError: Callback functions need to take 4 arguments: the
                iteration number, the width, the function that generates `y`
                values and the Population. Instead, a function was provided
                that takes (?) arguments. """
        # Check if function is `None` or callable.
        if not callable(function) and function is not None:
            raise TypeError("Callbacks can only be callable objects or "
                            "`None`. Provided was {}.".format(function))
        # If indeed a callable, check that the signature matches the expected
        # signature
        if callable(function):
            sig = signature(function)
            if len(sig.parameters) != 4:
                raise ValueError("Callback functions need to take 4 "
                                 "arguments: the iteration number, the "
                                 "width, the function that generates `y` "
                                 "values and the Population. Instead, a "
                                 "function was provided that takes {} "
                                 "arguments.".format(len(sig.parameters)))
        return function

    def validate_sampler(self, sampler):
        """ Validates that the provided sampler function `sampler` is a
        callable that takes two arguments.

        Args:
            sampler: Object to be validated as a valid sampler function.

        Returns:
            The handle to the sampler function that was provided as input.

        Raises:
            TypeError: Samplers need to be functions, but provided was '(?)',
                 a(n) (?). See the `samplers` module for example sampler
                 functions.
            ValueError: Sampler functions need to take two arguments: the
                number of points the sample `n` and the ranges within which
                these samples must lie. The provided sampler function takes (?)
                arguments instead. See the `samplers` module for example
                sampler functions."""
        if not callable(sampler):
            raise TypeError("Samplers need to be functions, but provided was "
                            "'{}', a(n) {}. See the `samplers` module for "
                            "example sampler functions.".format(
                                sampler, type(sampler)))
        sig = signature(sampler)
        if len(sig.parameters) != 2:
            raise ValueError("Sampler functions need to take two arguments: "
                             "the number of points the sample `n` and the "
                             "ranges within which these samples must lie. The "
                             "provided sampler function takes {} arguments "
                             "instead. See the `samplers` module for example "
                             "sampler functions.".format(len(sig.parameters)))
        return sampler

    def validate_emptiness(self):
        """ Validates if there is indeed no data currently stored in the
        Particle Filter. If there is, an error will be raised.

        Raises:
            Exception: There is already a seed defined. To define a new seed,
                you first have to reset the `ParticleFilter` object."""
        if self.population:
            raise Exception("There is already a seed defined. To define a new "
                            "seed you first have to reset the "
                            "`ParticleFilter` object.")

    def validate_function(self, function):
        """ Validates the function used to assign labels `y` to samples. This
        function should take only one parameter and should be a callable. If
        one of these assumptions is broken by the provided function, an
        Error is raised.

        Args:
            function: Object to be tested for being a valid labeling function.

        Returns:
            The handle to the function provided as input.

        Raises:
            TypeError:"""
        if not callable(function):
            raise TypeError("Labeling functions need to be callable, but "
                            "provided was '{}', a(n) {}.".format(
                                function, type(function)))
        sig = signature(function)
        if len(sig.parameters) != 1:
            raise ValueError("Labeling functions need to take one arguments: "
                             "the coordindates `x` for which the labels `y` "
                             "need to be determined. The provided function "
                             "takes {} arguments instead.".format(
                                 len(sig.parameters)))
        return function

    """ Callbacks """
    # To make the ParticleFilter as transparent as possible, a large collection
    # of callback functions can be provided. The methods in this section
    # implement the correct functionality of these callbacks.

    def _init_callbacks_dictionary(self):
        """ Initialises the internal dictionary in which handles to callbacks
        will be stored. Use the `add_callback` and `callback` methods to add
        and run these callbacks."""
        self._callbacks = {
            'after_population_init': [],
            'at_start_of_run': [],
            'at_start_of_iteration': [],
            'before_width_controller_call': [],
            'after_width_controller_call': [],
            'before_selection': [],
            'after_selection': [],
            'before_stdev_controller_call': [],
            'after_stdev_controller_call': [],
            'before_procreation': [],
            'after_procreation': [],
            'before_kill_controller_call': [],
            'after_kill_controller_call': [],
            'before_kill_data': [],
            'at_end_of_iteration': [],
        }

    def add_callback(self, name, function=None):
        """ Adds a callback function (or resets a callback function).

        The ParticleFilter class implements a variety of callback handles
        which can be filled by the user with callables through this method. See
        the docstring for the `validate_callback_name` for a list of
        implemented callback handles. See the docstring of the
        `validate_callback` method for more information on the signature
        callback functions are required to have.

        Callbacks are also allowed to be `None`, which is equivalent to there
        being no callback function for the provided handle.

        Args:
            name: Name of the callback, defining when the callback will be
                called (or which callbacks need to be removed, if `function`
                was set to `None`).
            function: A callable or `None`, defining the function that should
                be called when the callback condition is satisfied. If `None`
                is supplied, the existing callback for the provided handle is
                removed. Default is `None`. """
        # Validate callback
        name = self.validate_callback_name(name)
        function = self.validate_callback(function)
        # Store handle in callback dictionary
        if function is None:
            self._callbacks[name] = []
        else:
            self._callbacks[name].append(function)

    def callback(self, name):
        """ Runs the callback with provided name if the name exists. See the
        docstring for the `validate_callback_name` for a list of implemented
        callback handles.

        If the requested callback is `None`, the request is ignored.

        Args:
            name: Name of the callback, defining which callback will be
                attempts call. """
        # Check if `name` exists as a callback
        name = self.validate_callback_name(name)
        # Run all callbacks associated with the provided name
        for cb in self._callbacks[name]:
            cb(self.iteration, self.width, self.function, self.population)

    """ Data and function methods """
    # Particle filters play with data: they sample new data, select interesting
    # data for a next generation etc. To make the core loop as clear as
    # possible, the functions in this section define core operations that can
    # be called in a single line. Keeps the code nice and clean ^^.

    def set_seed(self, x, y):
        """ Creates a new `Population` instance and stores the provided
        coordinates `x` and function values `y` in it as seed.

        Args:
            x: numpy.ndarray containing the coordinates of the samples that
                need to be added to the `Population` object. The shape
                of each entry in this array should match the shape of the
                already stored entries.
            y: numpy.ndarray containing the function values for the coordinates
                provided as `x`. Each entry should have just one function
                value."""
        # Validate that there is no data already stored in the ParticleFilter
        self.validate_emptiness()
        # Create a Population object and store the provided coorindates `x`
        # and the function values `y` in it as seed.
        self.population = Population(x, y)
        self.callback("after_population_init")

    def sample_seed(self, n, sampler=samplers.uniform_sampler):
        """ Samples `n` datapoints with the provided `sampler` method.

        The ranges within the samples are taken is taken as the `boundaries`
        property of the `ParticleFilter` object.

        After sampling, the function values of the samples are evaluated using
        the `function` defined at initialisation of the `ParticleFilter` object
        and both the samples `x` and the function values `y` are stored as a
        seed in a `Population` object.

        Args:
            n: `int` defining the number of samples to take.
            sampler: Sampler function from the `samplers` module. Default is
                `samplers.uniform`. """
        # Validate that there is no data already stored in the ParticleFilter
        # and validate the provided sampler
        self.validate_emptiness()
        self.validate_sampler(sampler)
        # Sample data with the sampler
        x = sampler(n, self.boundaries)
        # Evaluate samples and store them with their function values
        y = self.function(x)
        self.set_seed(x, y)

    def set_boundaries(self, boundaries=None):
        """ Sets the boundaries within all samples need to fall.

        Boundaries can be provided as a `numpy.ndarray` or as `None`. If
        `None` is provided, it is interpreted as there being no boundaries to
        speak of, resulting in sampling ranges from `-inf_replace` to
        `+inf_replace`.

        Args:
            boundaries: The boundaries of the region within all samples need
                to be taken. See the docstring of `utils.validate_boundaries`
                for more information on what the requirements are on the
                formatting of boundaries. """
        boundaries = utils.validate_boundaries(boundaries)
        self.boundaries = boundaries

    """ Core methods """
    # The methods defined in this section contain the core functionality of
    # the particle filter. They include methods to start runs and to run
    # an iteration, but also to select data for procreation.

    # TODO: Maybe it is nice to have this method also in a controller-like
    #       structure? As it is now there is no choice possible from the user
    #       on how to determine the procreation rates, which is kinda rigid...
    def _calculate_procreation_rates(self, values, iteration_size):
        """ Calculates the procreation rates of data points based on their
        function value.

        The procreation rate is determined by defining a procreation
        probability based on the function values: the best (i.e. lowest)
        function value has the highest probability to procreate, the worst
        function value has a probability of 0 to procreate. This array of
        probabilities is then multiplied by the iteration size to get the
        array of procreation rates (after fixing for rounding errors).

        Args:
            values: `numpy.ndarray` of functional values of the data points for
                which the procreation rates have to be determined.
            iteration_size: integer representing the number of data points that
                need to be sampled as a result of the procreation rates
                calculated with this method.

        Returns:
            `numpy.ndarray` of `numpy.int32`s indicating for each of the
            data points how many offspring points it will have. """
        # Calculate probabilities for samples
        z = values - np.amin(values)
        if len(np.unique(z)) != 1:
            z = 1 - (z / np.amax(z))
            probabilities = z / np.sum(z)
        else:
            probabilities = np.ones(len(z)) / len(z)
        probabilities = probabilities.flatten()
        # Sort samples based on probability, from low to high
        idx = np.argsort(probabilities)[::-1]
        # Determine samples per point
        procreation_rates = np.ceil(probabilities * iteration_size)
        # Correct for rounding errors
        error = int(np.sum(procreation_rates) - iteration_size)
        if error > 0:
            n_zeros = np.sum(procreation_rates == 0)
            correction = np.zeros(len(procreation_rates))
            if n_zeros == 0:
                correction[-error:] = 1.0
            else:
                correction[-(error + n_zeros):-n_zeros] = 1.0
            procreation_rates[idx] = procreation_rates[idx] - correction
        # Return procreation rate
        return procreation_rates.astype(np.int32)

    def initialise_run(self):
        """ Start the run by initialising the iteration counter. """
        self.iteration = 0
        self.callback("at_start_of_run")

    def run_iteration(self):
        """ This method runs an interation of the particle filter algorithm.

        In an iteration, the following steps are taken in order:
        1.  Update the width using the `width_controller`;
        2.  Calculate procreation rates using `calculate_procreation_rates`;
        3.  Calculate the standard deviations for the normal distributions with
            the provided `stdev_controller`;
        4a. Sample new points
        4b. Evaluate their function values using the `function` provided at
            initialisation of the `ParticleFilter` object.
        4c. Store the samples and their function values.
        5.  Determine which data points should be removed from the population
            with the `kill_controller`.
        6a. Remove the points on the kill list, together with their function
            values, from the ParticleFilter.
        6b. End iteration. """
        self.iteration += 1
        self.callback('at_start_of_iteration')
        # Update the width using the width_controller
        self.callback('before_width_controller_call')
        self.width = self.width_controller(self.iteration - 1, self.width)
        self.callback('after_width_controller_call')
        # Calculate procreation rate
        self.callback('before_selection')
        x, y = self.population.get_data()
        rates = self._calculate_procreation_rates(y, self.iteration_size)
        self.population.set_procreation_rates(rates)
        self.callback('after_selection')
        # Calculate standard deviations of procreating data using the
        # stdev_controller
        self.callback('before_stdev_controller_call')
        x, _ = self.population.get_procreating_data()
        stdevs = self.stdev_controller(self.width, self.boundaries, x)
        self.population.set_stdevs(stdevs)
        self.callback('after_stdev_controller_call')
        # Let the points in the population procreate and evaluate their
        # function values using `function`. Store the results in the
        # Population.
        self.callback('before_procreation')
        x = self.population.procreate(self.boundaries,
                                      self.max_resample_attempts)
        y = self.function(x)
        self.population.append(x, y)
        self.callback('after_procreation')
        # Determine which points are killed at the end of this iteration using
        # the kill controller
        self.callback('before_kill_controller_call')
        _, y, origin = self.population.get_data_with_origin_information()
        is_new = (origin == self.iteration)
        kill_list = self.kill_controller(y, is_new, self.iteration_size)
        self.population.set_kill_list(kill_list)
        self.callback('after_kill_controller_call')
        # Kill data and end the iteration
        self.callback('before_kill_data')
        self.population.end_iteration()
        self.callback('at_end_of_iteration')
