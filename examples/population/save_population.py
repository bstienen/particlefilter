"""
Example particlefilter code: population/save_population.py
----------------------------------------------------------
It can be convenient to store the population at the end of a particle filter
run or iteration. Such operations can be implemented using callbacks (see
the examples in the callbacks folder for examples on that), but here we 
showcase how to save the population in the first place.
"""

import numpy as np
import particlefilter as pf

# Let's make some data to put in the population as a seed
n = 1000
x = np.random.rand(n, 2)
y = np.sum(np.power(x, 2), axis=1)

# Let's make a population 
population = pf.Population(x, y)

# To store the population, we need to define a location to store the
# information to. Storage format is .csv, so let's just store it as
# 'population.csv' in the current folder.
filepath = "./population.csv"
population.save(filepath)