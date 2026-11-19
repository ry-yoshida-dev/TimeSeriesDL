from .gaussian import GaussianObjective
from .laplace import LaplaceObjective
from .point import PointObjective
from .quantile import QuantileObjective
from .student_t import StudentTObjective

__all__ = ["GaussianObjective", "LaplaceObjective", "PointObjective", "QuantileObjective", "StudentTObjective"]
