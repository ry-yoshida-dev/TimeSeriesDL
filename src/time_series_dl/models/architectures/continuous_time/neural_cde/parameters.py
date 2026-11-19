from dataclasses import dataclass, field

from ....components.differential_equations import CubicInterpolationKind, ODESolverKind
from ....components.irregular_heads import IrregularEncoderHead
from ....interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....interfaces.shapes.irregular_summary import IrregularSummaryShape
from .irregular_sequence_encoder import NeuralCDESequenceEncoder


@dataclass(frozen=True, kw_only=True)
class NeuralCDEParameters:
    """
    Neural CDE (Kidger et al., 2020): a neural controlled differential equation driven by a cubic interpolation of
    the observations with observational intensity, followed by the output layers of ``head``. Defaults follow the
    paper's PhysioNet sepsis experiment with observational intensity, with the Hermite cubic with backward
    differences that the authors later recommend in place of the natural cubic spline.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden state.
    field_hidden_size : int
        Size of the hidden layers of the vector field.
    field_layers : int
        Number of hidden layers of the vector field.
    interpolation : CubicInterpolationKind
        Interpolation of the control path.
    solver : ODESolverKind
        Fixed-step solver.
    steps_per_interval : int
        Number of solver steps between consecutive observations.
    head : IrregularEncoderHead
        Output layers; a linear readout by default.

    Raises
    ------
    ValueError
        If a size or steps_per_interval is not positive.
    """

    hidden_size: int = 49
    field_hidden_size: int = 49
    field_layers: int = 4
    interpolation: CubicInterpolationKind = CubicInterpolationKind.HERMITE
    solver: ODESolverKind = ODESolverKind.RUNGE_KUTTA
    steps_per_interval: int = 1
    head: IrregularEncoderHead = field(default_factory=IrregularEncoderHead)

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_size", self.hidden_size),
            ("field_hidden_size", self.field_hidden_size),
            ("field_layers", self.field_layers),
            ("steps_per_interval", self.steps_per_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def build_encoder(self, input_channels: int) -> NeuralCDESequenceEncoder:
        """
        Build the untrained encoder.

        Parameters
        ----------
        input_channels : int
            Number of channels of the sequences.

        Returns
        -------
        NeuralCDESequenceEncoder
            The encoder.
        """
        return NeuralCDESequenceEncoder(
            input_channels,
            self.hidden_size,
            self.field_hidden_size,
            self.field_layers,
            self.interpolation.build(),
            self.solver.build(),
            self.steps_per_interval,
        )

    def build_irregular_summary_network(self, shape: IrregularSummaryShape) -> IrregularSummaryNetwork:
        """
        Build an untrained irregular summary network.

        Parameters
        ----------
        shape : IrregularSummaryShape
            The input and output dimensions.

        Returns
        -------
        IrregularSummaryNetwork
            The network.
        """
        return self.head.build_summary_network(self.build_encoder(shape.input_channels), shape)
