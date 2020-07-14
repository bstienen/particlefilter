"""
Example particlefilter code: basic/multiple_global_minima.py
---------------------------------- -------------------------
This code shows that the particle filter algorithm is able to find multiple
local minima, but also that it might struggle with finding *all* minima if the
default kill_controller is used. The reason for this is that this default
controller does not take the distance between the found points into account
and it might therefore happen that all particles in one of the minima are 
deleted.
"""

import numpy as np
import matplotlib.pyplot as plt
import particlefilter as pf

# Define the function that will be optimised using the particle filter. We use
# a 2-dimensional function with multiple global minima: sin(x)*cos(y).
def func(x):
    # Shift the input to make the result non trivial (i.e. the minimum is not
    # at the origin)
    # Return polynomial
    return np.sin(x[:, 0])*np.cos(x[:, 1])

# As this function is defined over the entire parameter space, we guide the
# sampling by the particle filter by defining boundaries within which we will
# be looking for the minima. Boundaries are defined through an array of shape
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

# That's all! We can now run the iterations of the particle filter, after
# initialising the run, with a simple for-loop and the `run_iteration` method
# of the particle filter. We set the number of iterations the algorithm should
# run to 100, but it is illustrative to change this number and see how the
# results of this code (see below) change!
n_iterations = 100
optimiser.initialise_run()
for iteration in range(n_iterations):
    optimiser.run_iteration()

# To visually *see* how the optimisation algorithm did, we plot the data from
# the latest iteration in a scatter plot (with function values indicated by
# their colour), together with the seed (drawn as crosses).
x, y = optimiser.population.get_data()
plt.scatter(x_seed[:, 0], x_seed[:, 1],
            c=y_seed, marker='x', label="Seed", alpha=0.5)
plt.scatter(x[:, 0], x[:, 1], c='k', s=10, label="Last iteration")
plt.plot([-10, -10, 10, 10, -10], [-10, 10, 10, -10, -10],
         'k--',label="boundaries")
plt.xlim([-12.5, 12.5])
plt.ylim([-12.5, 12.5])
plt.colorbar()
plt.legend()
plt.show()
