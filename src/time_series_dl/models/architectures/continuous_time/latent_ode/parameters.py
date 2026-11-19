from dataclasses import dataclass, field

from ....components.differential_equations import ODESolverKind
from ....components.irregular_heads import IrregularEncoderHead
from ....interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from ....interfaces.irregular_summary_network import IrregularSummaryNetwork
from ....interfaces.shapes.irregular_sequence import IrregularSequenceShape
from ....interfaces.shapes.irregular_summary import IrregularSummaryShape
from ..ode_rnn import ODERNNRecurrence
from .generative_model import LatentODEGenerativeModel
from .irregular_interpolation_network import LatentODEInterpolationNetwork
from .irregular_summary_network import LatentODESummaryNetwork


@dataclass(frozen=True, kw_only=True)
class LatentODEParameters:
    """
    Latent ODE (Rubanova et al., 2019): a variational model whose initial latent state follows a neural ordinary
    differential equation, inferred by an ODE-RNN read backward in time. Classification and regression read the
    initial latent state with the output layers of ``head``; interpolation and forecasting decode it at the query
    times. Defaults follow the authors' PhysioNet settings.

    Attributes
    ----------
    latent_size : int
        Size of the latent state.
    recognition_size : int
        Size of the hidden state of the recognition ODE-RNN.
    recognition_hidden_sizes : tuple[int, ...]
        Hidden layer sizes of the tanh network of the recognition ODE-RNN's dynamics.
    dynamics_hidden_sizes : tuple[int, ...]
        Hidden layer sizes of the tanh network of the latent dynamics.
    recognition_solver : ODESolverKind
        Fixed-step solver of the recognition ODE-RNN.
    dynamics_solver : ODESolverKind
        Fixed-step solver of the latent dynamics.
    steps_per_interval : int
        Number of solver steps between consecutive times, in both solvers.
    observation_std : float
        Standard deviation of the Gaussian likelihood of the observed values.
    kl_weight : float
        Weight of the Kullback-Leibler divergence in the evidence lower bound.
    output_loss_weight : float
        Weight of the classification or regression loss relative to the evidence lower bound.
    head : IrregularEncoderHead
        Output layers of the summary network.

    Raises
    ------
    ValueError
        If a size, steps_per_interval, observation_std or output_loss_weight is not positive, or kl_weight is
        negative.
    """

    latent_size: int = 20
    recognition_size: int = 40
    recognition_hidden_sizes: tuple[int, ...] = (50, 50, 50, 50)
    dynamics_hidden_sizes: tuple[int, ...] = (50, 50, 50, 50)
    recognition_solver: ODESolverKind = ODESolverKind.EULER
    dynamics_solver: ODESolverKind = ODESolverKind.DORMAND_PRINCE
    steps_per_interval: int = 2
    observation_std: float = 0.01
    kl_weight: float = 1.0
    output_loss_weight: float = 100.0
    head: IrregularEncoderHead = field(default_factory=lambda: IrregularEncoderHead(hidden_sizes=(300, 300)))

    def __post_init__(self) -> None:
        for name, value in (
            ("latent_size", self.latent_size),
            ("recognition_size", self.recognition_size),
            ("steps_per_interval", self.steps_per_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, sizes in (
            ("recognition_hidden_sizes", self.recognition_hidden_sizes),
            ("dynamics_hidden_sizes", self.dynamics_hidden_sizes),
        ):
            if any(size <= 0 for size in sizes):
                raise ValueError(f"{name} must be positive, got {sizes}")
        if self.observation_std <= 0.0:
            raise ValueError(f"observation_std must be positive, got {self.observation_std}")
        if self.kl_weight < 0.0:
            raise ValueError(f"kl_weight must not be negative, got {self.kl_weight}")
        if self.output_loss_weight <= 0.0:
            raise ValueError(f"output_loss_weight must be positive, got {self.output_loss_weight}")

    def build_generative_model(self, channels: int) -> LatentODEGenerativeModel:
        """
        Build the untrained generative model and its recognition network.

        Parameters
        ----------
        channels : int
            Number of channels of the sequences.

        Returns
        -------
        LatentODEGenerativeModel
            The model.
        """
        recognition: ODERNNRecurrence = ODERNNRecurrence(
            channels,
            self.recognition_size,
            self.recognition_hidden_sizes,
            self.recognition_solver.build(),
            self.steps_per_interval,
        )
        return LatentODEGenerativeModel(
            channels=channels,
            latent_size=self.latent_size,
            recognition=recognition,
            dynamics_hidden_sizes=self.dynamics_hidden_sizes,
            solver=self.dynamics_solver.build(),
            steps_per_interval=self.steps_per_interval,
            observation_std=self.observation_std,
            kl_weight=self.kl_weight,
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
        return LatentODESummaryNetwork(
            self.build_generative_model(shape.input_channels),
            self.head.build_output_layers(self.latent_size, shape.output_size),
            self.output_loss_weight,
        )

    def build_irregular_interpolation_network(self, shape: IrregularSequenceShape) -> IrregularInterpolationNetwork:
        """
        Build an untrained irregular interpolation network.

        Parameters
        ----------
        shape : IrregularSequenceShape
            The dimensions of the sequences.

        Returns
        -------
        IrregularInterpolationNetwork
            The network.
        """
        return LatentODEInterpolationNetwork(self.build_generative_model(shape.channels))
