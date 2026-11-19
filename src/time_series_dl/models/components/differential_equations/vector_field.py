"""Type of the right-hand side of an ordinary differential equation."""

from collections.abc import Callable

from torch import Tensor

type VectorField = Callable[[Tensor, Tensor], Tensor]
"""Maps a time and a state to the time derivative of the state, with the shape of the state."""
