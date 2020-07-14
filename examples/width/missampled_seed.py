"""
Example particlefilter code: width/missampled_seed.py
-----------------------------------------------------
In the width/width_decay.py example it was stated that the width and
width_decay are important parameters of the particle filter algorithm. That
this is indeed the case, is shown in this script. Here we artificially
missample the seed and put the initial width purposfully very small. This
results in the algorithm not being able to find the global minimum, but getting
stuck in local minima.
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
# Additionally we define boundaries specifically for the seed, so that it will
# be missampled with respect to the global minimum
seed_boundaries = np.array([[7, 10], [7, 10]])

# We create an initial seed from which the particle filter iterations will
# start. To this end, we sample random coordinates `x` (within the 
# seed_boundaries defined above) and evaluate the function value for these
# coordinates.
n_seed = 100
dx = seed_boundaries[:, 1] - seed_boundaries[:, 0]
x_seed = np.random.rand(n_seed, 2)*dx + seed_boundaries[:, 0]
y_seed = func(x_seed)

# For this example we set the initial width and the width_decay purposfully
# very low. Play with this yourself and see what might work and what does not! 
initial_width = 0.2
width_decay = 0.75
width_controller = pf.get_width_controller(decay_rate=width_decay,
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
    print(optimiser.width)

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
            c=y_seed, marker='x', label="Seed", alpha=0.5, vmin=-1.0)
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
