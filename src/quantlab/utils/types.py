"""Module for common types definition."""
from typing import Union

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Floats = Union[float, FloatArray]
