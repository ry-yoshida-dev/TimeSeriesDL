from dataclasses import dataclass, field

from ....components.differential_equations import ODESolverKind
from ....components.irregular_heads import IrregularEncoderHead
from ....interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....interfaces.shapes.irregular_summary import IrregularSummaryShape
from .irregular_sequence_encoder import ODERNNSequenceEncoder
from .recurrence import ODERNNRecurrence


@dataclass(frozen=True, kw_only=True)
class ODERNNParameters:
    """
    ODE-RNN (Rubanova et al., 2019): a GRU whose hidden state follows a neural ordinary differential equation between
    observations, followed by the output layers of ``head``. Defaults follow the authors' PhysioNet classification
    settings.

    Attributes
    ----------
    hidden_size : int
        Size of the hidden state.
    ode_hidden_sizes : tuple[int, ...]
        Hidden layer sizes of the tanh network that defines the derivative of the hidden state.
    solver : ODESolverKind
        Fixed-step solver.
    steps_per_interval : int
        Number of solver steps between consecutive observations.
    head : IrregularEncoderHead
        Output layers.

    Raises
    ------
    ValueError
        If a size or steps_per_interval is not positive.
    """

    hidden_size: int = 20
    ode_hidden_sizes: tuple[int, ...] = (50, 50, 50, 50)
    solver: ODESolverKind = ODESolverKind.EULER
    steps_per_interval: int = 2
    head: IrregularEncoderHead = field(default_factory=lambda: IrregularEncoderHead(hidden_sizes=(300, 300)))

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if any(size <= 0 for size in self.ode_hidden_sizes):
            raise ValueError(f"ode_hidden_sizes must be positive, got {self.ode_hidden_sizes}")
        if self.steps_per_interval <= 0:
            raise ValueError(f"steps_per_interval must be positive, got {self.steps_per_interval}")

    def build_recurrence(self, input_channels: int) -> ODERNNRecurrence:
        """
        Build the untrained ODE-RNN.

        Parameters
        ----------
        input_channels : int
            Number of channels of the sequences.

        Returns
        -------
        ODERNNRecurrence
            The ODE-RNN.
        """
        return ODERNNRecurrence(
            input_channels, self.hidden_size, self.ode_hidden_sizes, self.solver.build(), self.steps_per_interval
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
        return self.head.build_summary_network(
            ODERNNSequenceEncoder(self.build_recurrence(shape.input_channels)), shape
        )
