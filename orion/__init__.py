"""
The Orion library is a Python package designed for training medical video models. It provides functionality for loading coronary angiogram videos and includes functions for training and testing segmentation and ejection fraction prediction models.
"""

import orion.datasets as datasets
import orion.models as models
import orion.utils as utils
from orion.__version__ import __version__

__all__ = ["__version__", "datasets", "utils"]
