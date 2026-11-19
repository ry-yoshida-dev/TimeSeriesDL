"""NumPy array aliases shared by the package."""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating]
type Float32Array = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]
type IndexArray = NDArray[np.intp]
type LabelScalar = np.integer | np.bool_ | np.str_
type LabelArray = NDArray[LabelScalar]
