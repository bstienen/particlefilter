"""
Example particlefilter code: stdevs/stdev_controller.py
-------------------------------------------------------
One of the more complex components of this implementation of the particle
filter is the standard deviation controller (`stdev_controller` for short).
This controller is a function takes in three arguments:

- `width`: value for the current width parameter
- `boundaries`: defines the range within which samples must fall
- `x`: the coordinates for which the standard deviations must be determined.

The function should then return a `numpy.ndarray` of shape `(nDatapoints, )`
containing the standard deviations. You can probabily imagine that the 
exact calculation of the standard deviations is highly customisable and that
the specific configuration chosen here can greatly impact the performance of
the particle filter.

Nevertheless and attempt has been made to implement a general std_controller 
within the particlefilter package. This controller can be obtained by calling
the `get_stdev_controller` function. This function (that, indeed, returns a
function) accepts arguments itself; arguments that configure the inner workings
of the controller it returns. This is technically implemented as a curry
function (in case you want to look up how it works), but in this script we will
just investigate how it *works*. To this end, we will not create a
`ParticleFilter` object, but instead just look at the controller itself and
see what it does.
"""

import numpy as np
import particlefilter as pf

# Before we begin, we define the parameters that the controller takes as input,
# so that we don't have to define them every time. In this example we will be
# looking at five 2-dimensional datapoints. The samples are bounded by a
# boundary at 0 and 10 for the first dimension and at 0 and 100 for the second.
width = 2.5
boundaries = np.array( [[0, 10], [-100, 100]])
x = np.arange(10).reshape(5, 2)

# Let's create the simplest controller possible: one that returns standard
# deviations for each dimension equal to the provided width parameter.
controller = pf.get_stdev_controller(scales_with_boundary=False)
stdevs = controller(width, boundaries, x)
print("SIMPLEST CONTROLLER")
print("Should be 2.5 for all dimensions and all 5 points:")
print(stdevs, end="\n\n")

# If we set `scales_width_boundary` to `True` however, the results are
# multiplied by the allowed sample ranges for each of the dimensions. As the
# ranges for the dimensions differ, this results in columns with different
# values. This is the default behaviour.
controller = pf.get_stdev_controller()
stdevs = controller(width, boundaries, x)
print("SCALES WITH BOUNDARY")
print("Should be 25 for the first column, 500 for the second")
print(stdevs, end="\n\n")

# Maybe we find a standard deviation of 500 to be a bit too high though, or
# 25 to be a bit too low. We can solve this by setting clip values. These
# clip the standard deviations to specific values if they are either lower or
# higher than a defined minimum or maximum.
controller = pf.get_stdev_controller(min_stdev=40.0, max_stdev=100.0)
stdevs = controller(width, boundaries, x)
print("CLIPPED STANDARD DEVIATIONS")
print("Should be 40 for the first column, 100 for the second")
print(stdevs, end="\n\n")

# Lastly: you can make the standard deviations depend on the location of the
# distribution, by setting the `logarithmic` argument to `True`. This
# multiplies the calculated stdevs (as a last step) by the locations `x`.
controller = pf.get_stdev_controller(scales_with_boundary=False,
                                     logarithmic=True)
stdevs = controller(width, boundaries, x)
print("LOGARITHMIC")
print("The locations `x` are: ")
print(x)
print("Should be 2.5 times the locations: ")
print(stdevs)
