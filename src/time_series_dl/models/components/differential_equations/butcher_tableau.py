import math
from dataclasses import dataclass
from typing import Final

_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, kw_only=True)
class ButcherTableau:
    """
    Coefficients of an explicit Runge-Kutta method.

    Attributes
    ----------
    nodes : tuple[float, ...]
        Fraction of the step at which each stage evaluates the vector field; the first is 0.
    coefficients : tuple[tuple[float, ...], ...]
        Weights of the earlier stages' derivatives in each stage's state; row ``i`` has ``i`` entries.
    weights : tuple[float, ...]
        Weights of the stages' derivatives in the step.

    Raises
    ------
    ValueError
        If the sizes disagree, the first node is not 0, a row does not sum to its node, or the weights do not sum
        to 1.
    """

    nodes: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        num_stages: int = len(self.nodes)
        if num_stages == 0 or len(self.coefficients) != num_stages or len(self.weights) != num_stages:
            raise ValueError(
                "nodes, coefficients and weights must have one entry per stage, got "
                + f"{len(self.nodes)}, {len(self.coefficients)} and {len(self.weights)}"
            )
        if self.nodes[0] != 0.0:
            raise ValueError(f"the first node must be 0, got {self.nodes[0]}")
        for stage, (node, row) in enumerate(zip(self.nodes, self.coefficients, strict=True)):
            if len(row) != stage:
                raise ValueError(f"coefficient row {stage} must have {stage} entries, got {len(row)}")
            if not math.isclose(sum(row), node, abs_tol=_TOLERANCE):
                raise ValueError(f"coefficient row {stage} must sum to its node {node}, got {sum(row)}")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=_TOLERANCE):
            raise ValueError(f"weights must sum to 1, got {sum(self.weights)}")

    @property
    def num_stages(self) -> int:
        """
        Get the number of vector field evaluations per step.

        Returns
        -------
        int
            The number of stages.
        """
        return len(self.nodes)
