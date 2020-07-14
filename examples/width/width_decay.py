"""
Example particlefilter code: width/width_decay.py
-------------------------------------------------
Two of the most important parameters in the particle filter are the width and
the width_decay. Together they define how quickly the particle filter will
zoom in on the minima that it found. This code showcases how you can configure
the width and the width_decay.
The function we will optimise in this example is the function from the example
in basic/multiple_local_minima.
"""

import numpy as np
import matplotlib.pyplot as plt
import particlefilter as pf

# Define the function that will be optimised using the particle filter. The
# function we use for this is defined at the top of this code.
location_minimum = np.array([-5, -5])
def func(x):
    # Shift the input to make the result non trivial (i.e. the minimum is not
    # at the origin)
    x = x - location_minimum
    # Define radius from (relocated) origin
    r = np.sqrt(np.sum(np.power(x, 2),axis=1))
    # Return function values
    return -np.cos(r) + 0.25*r

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

# Configuring the initial width is as easy as setting it equal to a specific
# floating point number. The default for this is 2.0, but for the sake of
# example, we will set it to something else than the default.
initial_width = 1.0
# The decay of this width is however a bit more involved. To control this, you
# need a function that takes two arguments: the current width and the current
# iteration number. It should return the new width for the current iteration.
# Although you can write such a function yourself, a function has been
# implemented in the `particlefilter` package that returns such a function for 
# you. This function-returning-a-function construction allows you to configure
# the specific decay rate and the frequency with which this decay rate should
# be applied (i.e. "it should apply this decay rate every N iterations", where
# n is configurable).
# Here we will use this implementation to get ourselves a valid width
# controller. It is illustrative (given the output we generate below) to play
# with the settings of the `get_width_controller` function.
width_controller = pf.get_width_controller(decay_rate=0.95,
                                           apply_every_n_iterations=1)

# With all this information available, we can initialise the particle filter
# object and set the seed equal to the data we just calculated.
optimiser = pf.ParticleFilter(function=func,
                              iteration_size=100,
                              boundaries=boundaries,
                              initial_width=initial_width,
                              width_controller=width_controller)
optimiser.set_seed(x_seed, y_seed)

# That's all! We can now run the iterations of the particle filter, after
# initialising the run, with a simple for-loop and the `run_iteration` method
# of the particle filter. At the end of each iteration we store the width
# used in that iteration, so that we can make nice plots afterwards.
n_iterations = 100
widths = [None] * n_iterations

optimiser.initialise_run()
for iteration in range(n_iterations):
    optimiser.run_iteration()
    widths[iteration] = optimiser.width

# Let's create a plot of the widths used for each of the epochs, to validate
# that it indeed decreased the width at the requested rate.
epochs = np.arange(n_iterations)
plt.scatter(epochs, widths)
plt.xlabel("Epoch")
plt.ylabel("Width")
plt.yscale("log")
plt.show()

# To visually *see* how the optimisation algorithm did, we plot the data from
# the latest iteration in a scatter plot (with function values indicated by
# their colour), together with the seed (drawn as crosses).
x, y = optimiser.population.get_data()
plt.scatter(x_seed[:, 0], x_seed[:, 1],
            c=y_seed, marker='x', label="Seed", alpha=0.5)
plt.scatter(x[:, 0], x[:, 1], c='k', s=10, label="Last iteration")
plt.scatter([location_minimum[0]], [location_minimum[1]],
            c='r', marker="*", label="Actual minimum")
plt.plot([-10, -10, 10, 10, -10], [-10, 10, 10, -10, -10],
         'k--',label="boundaries")
plt.xlim([-12.5, 12.5])
plt.ylim([-12.5, 12.5])
plt.colorbar()
plt.legend()
plt.show()

# We are however interested in the minimum, so let's print the point (and the
# function value) for the smallest point found.
idx = np.argmin(y)
dist = np.sqrt(np.sum(np.power(x[idx]-location_minimum, 2)))
print("Found minimum {} at {}".format(y[idx][0], x[idx]))
print("Euclidean distance to actual minimum is {}".format(round(dist, 8)))
