# Callbacks

To make the particle filter algorithm as customisable and transparent as possible, the `ParticleFilter` class implements to possibility to run callback functions at a wide selection of points in the run and iterations. The `adding_callbacks.py` example shows how you can add callbacks and the  `running_with_callbacks.py` example shows an example run with a selection of callbacks. 

## Signature of a callback
A callback function should take exactly four arguments.

- `iteration`: the iteration number
- `width`: the value for the width parameter
- `function`: the callable to be optimised
- `population`: the `Population` object that contains all data

With this information we can for example write a callback that prints the width parameter:

    def callback_function(iteration, width, function, population):
        print("Iteration {} has width parameter {}".format(iteration, width))

## Adding a callback
To add a callback, you run the `add_callback` method on your `ParticleFilter` object. This method takes two arguments: the location at which your callback should be called (see below) and the handle of your callback function. So for example:

    pf.add_callback('at_start_of_run', callback_function)

## Possible callback locations.

Below you can find a list of locations to which you can assign callback functions. This list is in order of appearance in a run:

- **`after_population_init`**: called immediately after the `Population` object has been created within the `ParticleFilter` object (i.e. after `set_seed` or `sample_seed`);
- **`at_start_of_run`**: called at `init_run`;
- **`at_start_of_iteration`**: called at the start of `run_iteration`. It is essentially the same as `before_width_controller_call`.
- **`before_width_controller_call`**: called before the width controller is run to update the `width` parameter. It is essentially the same as `at_start_of_iteration`.
- **`after_width_controller_call`**: called after the width controller is run to update the `width` parameter. It is essentially the same as `before_selection`.
- **`before_selection`**: called before the procreation rate of each point is determined. It is essentially the same as `after_width_controller_call`.
- **`after_selection`**: called right after the procreation rate of each point is determined. It is essentially the same as `before_stdev_controller_call`.
- **`before_stdev_controller_call`**: called before the standard deviations of the population is calculated with the `stdev_controller`. It is essentially the same as `after_selection`.
- **`after_stdev_controller_call`**: called right after the standard deviations of the population is calculated with the `stdev_controller`. It is essentially the same as `before_procreation`.
- **`before_procreation`**: called before new datapoints are sampled. It is essentially the same as `after_stdev_controller_call`.
- **`after_procreation`**: called right after newly sampled and evaluated points are added to the population. It is essentially the same as `before_kill_controller_call`.
- **`before_kill_controller_call`**: called before the kill controller, that determines which points are removed at the end of the iteration, is called. It is essentially the same as `after_procreation`.
- **`after_kill_controller_call`**: called right after the kill controller, that determines which points are removed at the end of the iteration, is called. It is essentially the same as `before_kill_data`.
- **`before_kill_data`**: called before the data marked for removal is removed from the population. It is essentially the same as `after_kill_controller_call`.
- **`at_end_of_iteration`**: called at the end of each iteration, i.e. after the data marked for removal is removed.