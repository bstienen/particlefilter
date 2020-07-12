# -*- coding: utf-8 -*-
import logging

from .__version__ import __version__  # flake8: noqa
from .particlefilter import ParticleFilter  # flake8: noqa
from .population import Population  # flake8: noqa
from .controllers import (get_width_controller, get_stdev_controller,
                          get_kill_controller)  # flake8: noqa
from .samplers import uniform_sampler  # flake8: noqa

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Bob Stienen"
__email__ = 'bstienen@science.ru.nl'
