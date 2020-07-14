"""
Example particlefilter code: callbacks/running_with_callbacks.py
----------------------------------------------------------------
In this code we run the same optimisation as in example basic/convex.py, but
now we define and set a series of callbacks that show how you could use them.
To learn how to write callbacks, see the callbacks/setting_callbacks.py
example.
"""

import numpy as np
import particlefilter as pf

# Define the function that will be optimised using the particle filter. We use
# a simple 2-dimensional second order polynomial in this example, but there
# is no reason why not a more complicated function could be used.
location_minimum = np.array([-5, -5])
def func(x):
    # Shift the input to make the result non trivial (i.e. the minimum is not
    # at the origin)
    x = x - location_minimum
    # Return polynomial
    return np.sum(np.power(x, 2), axis=1)

# As this function is defined over the entire parameter space, we guide the
# sampling by the particle filter by defining boundaries within which we will
# be looking for the minimum. Boundaries are defined through an array of shape
# `(nDimensions, 2)`, where the last axis contains the lower and upper
# boundary (in that order) for each of the dimensions.
boundaries = np.array([[-10, 10], [-10, 10]])

# We create an initial seed from which the particle filter iterations will
# start. To this end, we sample random coordinates `x` (within the boundaries
# defined above) and evaluate the function value for these coordinates.
n_seed = 100
dx = boundaries[:, 1] - boundaries[:, 0]
x_seed = np.random.rand(n_seed, 2)*dx + boundaries[:, 0]
y_seed = func(x_seed)

# With all this information available, we can initialise the particle filter
# object and set the seed equal to the data we just calculated.
optimiser = pf.ParticleFilter(function=func,
                              iteration_size=100,
                              boundaries=boundaries)
optimiser.set_seed(x_seed, y_seed)

# Let us define the callback functions that we want to have run during the
# particle filter run
def print_iteration_number(iteration, width, function, population):
    print("ITERATION {}".format(iteration))

def print_population_size(iteration, width, function, population):
    print("Population size: {}".format( len(population) ))

def store_population(iteration, width, function, population):
    population.save("population_{}.csv".format(iteration))

# Add these callbacks to the particle filter. The README in the callback
# examples folder lists which handles you can define.
optimiser.add_callback("at_start_of_iteration", print_iteration_number)
optimiser.add_callback("after_kill_data", store_population)
optimiser.add_callback("at_end_of_iteration", print_population_size)
optimiser.add_callback("at_end_of_iteration", store_population)

# That's all! We can now run the iterations of the particle filter, after
# initialising the run, with a simple for-loop and the `run_iteration` method
# of the particle filter. We set the number of iterations the algorithm should
# run to 100, but it is illustrative to change this number and see how the
# results of this code (see below) change!
n_iterations = 10
optimiser.initialise_run()
for iteration in range(n_iterations):
    optimiser.run_iteration()

# We are however interested in the minimum, so let's print the point (and the
# function value) for the smallest point found.
x, y = optimiser.population.get_data()
idx = np.argmin(y)
dist = np.sqrt(np.sum(np.power(x[idx]-location_minimum, 2)))
print("Found minimum {} at {}".format(y[idx][0], x[idx]))
print("Euclidean distance to actual minimum is {}".format(round(dist, 8)))
