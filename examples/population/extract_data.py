"""
Example particlefilter code: population/extract_data.py
-------------------------------------------------------
The `Population` object is the smart data container used in the particle
filter package. It stores all information associated with the currently
alive data points, like standard deviations and the procreation rate. Although
the storage properties of `Population` objects are not marked as private (for
as far as that is possible in python), it is recommended to use the data
retrieval API over direct accessing information through the properties. This
code shows the available methods to do this.
"""

import numpy as np
import particlefilter as pf

# Let us first define functions with which we can quickly create some data.
def func(x):
    return np.sum(x, axis=1)

def make_data(n):
    x = np.random.rand(n, 2)
    y = func(x)
    return (x, y)

# Let's now make some data and add it to the population. We also add a second
# batch of data.
x, y = make_data(100)
x2, y2 = make_data(50)
population = pf.Population(x, y)
population.append(x2, y2)

# The population is now filled with a seed (added in iteration 0) and an
# additional batch of data (added in iteration 1, which is still 'running' by
# the by). What is missing for this data is information on the procreation
# rates, the standard deviations and the kill list information. In a run of the
# particle filter this information is supplied to the `Population` object by 
# the parent `ParticleFilter` object. To fully explore the `Population`
# functionality, we simulate this behaviour.
population.set_procreation_rates( np.round(np.random.rand(150)) )
population.set_kill_list( np.random.rand(150) > 0.7 )

# Below all available data retrieval methods are listed.
# First: getting the number of data points in the population is as easy as
# requesting the length of the population:
n = len(population)  # = 150

# We can also get the current extremes of the function values y
minimum, maximum = population.get_function_extremes()

# Get all data coordinates and function values that are currently in the
# population
x, y = population.get_data()

# Get all data coordinates, function values and array of iterations of origin
# currently in the population
x, y, origin = population.get_data_with_origin_information()

# For sampling it is convenient to get the coorindates and the standard
# deviations from the population. As the coordinates will be the mean of the
# normal distributions from which we will sample, we can get this information
# with the following method
mean, stdevs = population.get_means_and_stdevs_for_sampling()

# If we only want to get the most recently added data:
x, y = population.get_latest_data()

# To get all data points that will be procreating the current iteration, we run
# the `get_procreating_data` method. This will return all data with a
# procreation rate of at least 1.
x, y = population.get_procreating_data()  # Will return approx. 75 points

# To get all data points that have been slated for deletion at the end of the
# current iteration, we request all data points on the 'kill list'.
x, y = population.get_data_on_kill_list()
