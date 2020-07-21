"""
Example particlefilter code: basic/convex.py
--------------------------------------------
The most basic of applications of the particle filter: a convex optimisation
problem. This code is almost as bare bones as it could be, were it not for the
overhead code present at the end file to validate the performance of the
algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
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

# That's all! We can now run the iterations of the particle filter, after
# initialising the run, with a simple for-loop and the `run_iteration` method
# of the particle filter. We set the number of iterations the algorithm should
# run to 100, but it is illustrative to change this number and see how the
# results of this code (see below) change!
n_iterations = 100
optimiser.initialise_run(graveyard_path="./graveyard.csv")
for iteration in range(n_iterations):
    optimiser.run_iteration()

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
