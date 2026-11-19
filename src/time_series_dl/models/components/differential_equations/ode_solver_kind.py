from enum import Enum

from .butcher_tableau import ButcherTableau
from .ode_solver import ODESolver


class ODESolverKind(Enum):
    """
    Fixed-step explicit Runge-Kutta method.

    Attributes
    ----------
    EULER
        Forward Euler, first order with one evaluation per step.
    RUNGE_KUTTA
        Classical Runge-Kutta, fourth order with four evaluations per step.
    DORMAND_PRINCE
        The fifth-order solution of Dormand-Prince (dopri5) with six evaluations per step, at a fixed step size
        instead of the adaptive one of ``torchdiffeq``.
    """

    EULER = "euler"
    RUNGE_KUTTA = "rk4"
    DORMAND_PRINCE = "dopri5"

    @property
    def tableau(self) -> ButcherTableau:
        """
        Get the coefficients of the method.

        Returns
        -------
        ButcherTableau
            The tableau.
        """
        match self:
            case ODESolverKind.EULER:
                return ButcherTableau(nodes=(0.0,), coefficients=((),), weights=(1.0,))
            case ODESolverKind.RUNGE_KUTTA:
                return ButcherTableau(
                    nodes=(0.0, 0.5, 0.5, 1.0),
                    coefficients=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
                    weights=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
                )
            case ODESolverKind.DORMAND_PRINCE:
                return ButcherTableau(
                    nodes=(0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0),
                    coefficients=(
                        (),
                        (1 / 5,),
                        (3 / 40, 9 / 40),
                        (44 / 45, -56 / 15, 32 / 9),
                        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
                        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
                    ),
                    weights=(35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
                )

    def build(self) -> ODESolver:
        """
        Build the solver.

        Returns
        -------
        ODESolver
            The solver.
        """
        return ODESolver(self.tableau)
