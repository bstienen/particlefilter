"""
Example particlefilter code: high_dimensionality/convex_animation.py
--------------------------------------------------------------------
This example code shows how the particle filter performs when the function to
optimise is high dimensional. Typically these are challening problems, as they
require a large seed and a large iteration size.
The function we will be optimising here will be the more complex function from
the basic/multiple_local_minima.py example:

    f(r) = - cos(r) + 0.25*r
        with r = np.sqrt( sum( x_i**2 ) )

This script requires the celluloid python package to be installed, as it also
outputs how the function values develop over the iterations. The
high_dimensionality/multiple_local_minima.py script contains the same code, but
does not make this animation.
"""

import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import particlefilter as pf

# Let's define the number of dimensions, the seed size and the iteration size
n_dimensions = 5
n_seed = 500
iteration_size = 500

# Define the function that will be optimised using the particle filter.
location_minimum = np.array([-5]*n_dimensions)

def get_distance_to_minimum(x):
    # Shift the input to make the result non trivial (i.e. the minimum is not
    # at the origin)
    x = x - location_minimum
    # Define radius from (relocated) origin
    return np.sqrt(np.sum(np.power(x, 2),axis=1))

def func(x):
    r = get_distance_to_minimum(x)
    # Return function values
    return -np.cos(r) + 0.25*r

# As this function is defined over the entire parameter space, we guide the
# sampling by the particle filter by defining boundaries within which we will
# be looking for the minimum. Boundaries are defined through an array of shape
# `(nDimensions, 2)`, where the last axis contains the lower and upper
# boundary (in that order) for each of the dimensions.
boundaries = np.array([[-10, 10]]*n_dimensions)

# We create an initial seed from which the particle filter iterations will
# start. To this end, we sample random coordinates `x` (within the boundaries
# defined above) and evaluate the function value for these coordinates.
dx = boundaries[:, 1] - boundaries[:, 0]
x_seed = np.random.rand(n_seed, n_dimensions)*dx + boundaries[:, 0]
y_seed = func(x_seed)

# As this example is a bit more complicated to optimise, we need to use
# a non-default width controller. See the width/width_decay.py example for
# more information (and extra plots).
initial_width = 1
width_decay = 0.99
width_controller = pf.get_width_controller(decay_rate=width_decay,
                                           apply_every_n_iterations=1)

def stdev_controller(width, boundaries, x):
    return 10*width*np.ones(len(x))

# With all this information available, we can initialise the particle filter
# object and set the seed equal to the data we just calculated.
# We need to set the width relatively low by the way, as otherwise a very large
# new samples will be lay outside the boundaries defined above. The
# probability that this happens increases with the dimensionality, so if you
# change the dimensionality at the top of this script, you might also want to
# change the initial width defined here.
optimiser = pf.ParticleFilter(function=func,
                              iteration_size=iteration_size,
                              boundaries=None,
                              initial_width=initial_width,
                              width_controller=width_controller,
                              stdev_controller=stdev_controller)
optimiser.set_seed(x_seed, y_seed)

# Before we run the iterations, we initialise the figure and camera object
# needed for the animation.
fig = plt.figure()
camera = Camera(fig)
hist_max = np.sqrt(15*15*n_dimensions)+1

# Let's run the optimisation.
n_iterations = 250
optimiser.initialise_run()
for iteration in range(n_iterations):
    print("Iteration {}".format(iteration))
    optimiser.run_iteration()
    # Create histogram
    x, _ = optimiser.population.get_data()
    print(len(x))
    plt.hist(get_distance_to_minimum(x), 20)
    plt.xscale('log')
    camera.snap()

# Store the animation as convex.mp4 in the current folder.
animation = camera.animate()
animation.save('multiple_local_minima.mp4')

# As we are however interested in the minimum, so let's print the point (and
# the function value) for the smallest point found.
x, y = optimiser.population.get_data()
print(x.shape)
idx = np.argmin(y)
dist = np.sqrt(np.sum(np.power(x[idx]-location_minimum, 2)))
print("Found minimum {} at {}".format(y[idx][0], x[idx]))
print("Euclidean distance to actual minimum is {}".format(round(dist, 8)))
