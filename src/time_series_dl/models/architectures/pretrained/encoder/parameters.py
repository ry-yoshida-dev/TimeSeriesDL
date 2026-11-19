from dataclasses import dataclass
from pathlib import Path

from .....runtime import Device
from ....components.heads import EncoderHead
from ....interfaces import (
    ForecastNetwork,
    ForecastShape,
    ReconstructionNetwork,
    SequenceEncoder,
    SequenceShape,
    SummaryNetwork,
    SummaryShape,
)
from ....interfaces.pretraining_network import PretrainingNetwork
from .frozen_sequence_encoder import FrozenSequenceEncoder


@dataclass(frozen=True, kw_only=True)
class PretrainedEncoderParameters:
    """
    The encoder of a fitted ``RepresentationLearner``, read from its saved file, with the generic heads of every
    task: fine-tuned with the head, or frozen so that only the head is trained (linear probing).

    The file is read whenever a network is built, including when a fitted task that uses these parameters is
    loaded (its saved weights then replace the pretrained ones), so it must stay at ``checkpoint_path``. The
    encoder is rebuilt for the length of the task's sequences and takes the pretrained weights, which works for
    any length when the encoder's weights do not depend on it (such as the convolutional TS2Vec and TS-TCC
    encoders), and only for the pretraining length otherwise. The number of channels must be that of pretraining.
    Tasks scale their inputs with their own statistics; keep the scaling of pretraining (standard scaling by
    default) so the encoder sees inputs like those it was trained on, and turn off the instance normalization of
    forecasting if pretraining did not use it.

    Attributes
    ----------
    checkpoint_path : str
        Path of the file written by ``RepresentationLearner.save``; relative paths are resolved against the
        working directory when a network is built.
    is_encoder_frozen : bool
        Whether the encoder's weights stay fixed while the task trains.
    head : EncoderHead | None
        Output layers of every task; when None, summaries pool the features like the instance-level
        representation of the pretraining method (such as the maximum over time of TS2Vec), without dropout.

    Raises
    ------
    ValueError
        If checkpoint_path is empty.
    """

    checkpoint_path: str
    is_encoder_frozen: bool = False
    head: EncoderHead | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_path.strip():
            raise ValueError("checkpoint_path must not be empty.")

    def build_encoder(self, input_length: int, input_channels: int) -> SequenceEncoder:
        """
        Build the pretrained encoder, frozen if requested.

        Parameters
        ----------
        input_length : int
            Number of time steps of the input.
        input_channels : int
            Number of channels of the input; must be that of pretraining.

        Returns
        -------
        SequenceEncoder
            The encoder with its pretrained weights.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a representation learner, or its encoder does not fit the input.
        """
        encoder: SequenceEncoder
        encoder, _ = self._build(input_length, input_channels)
        return encoder

    def build_forecast_network(self, shape: ForecastShape) -> ForecastNetwork:
        """
        Build a forecast network around the pretrained encoder.

        Parameters
        ----------
        shape : ForecastShape
            The input and output dimensions.

        Returns
        -------
        ForecastNetwork
            The network.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a representation learner, or its encoder does not fit the input.
        """
        encoder: SequenceEncoder
        head: EncoderHead
        encoder, head = self._build(shape.input_length, shape.input_channels)
        return head.build_forecast_network(encoder, shape)

    def build_reconstruction_network(self, shape: SequenceShape) -> ReconstructionNetwork:
        """
        Build a reconstruction network around the pretrained encoder.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the sequences.

        Returns
        -------
        ReconstructionNetwork
            The network.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a representation learner, its encoder does not fit the input, or it changes the
            length of the sequence.
        """
        encoder: SequenceEncoder
        head: EncoderHead
        encoder, head = self._build(shape.length, shape.channels)
        return head.build_reconstruction_network(encoder, shape)

    def build_summary_network(self, shape: SummaryShape) -> SummaryNetwork:
        """
        Build a summary network around the pretrained encoder.

        Parameters
        ----------
        shape : SummaryShape
            The input and output dimensions.

        Returns
        -------
        SummaryNetwork
            The network.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a representation learner, or its encoder does not fit the input.
        """
        encoder: SequenceEncoder
        head: EncoderHead
        encoder, head = self._build(shape.input_length, shape.input_channels)
        return head.build_summary_network(encoder, shape)

    def _build(self, input_length: int, input_channels: int) -> tuple[SequenceEncoder, EncoderHead]:
        # Imported here because the task imports the models; the learner owns the file format and rebuilds the
        # pretraining network of any method.
        from .....tasks.representation_learning import RepresentationLearner

        learner: RepresentationLearner = RepresentationLearner.load(Path(self.checkpoint_path), device=Device.CPU)
        network: PretrainingNetwork = learner.build_transfer_network(input_length, input_channels)
        encoder: SequenceEncoder = FrozenSequenceEncoder(network.encoder) if self.is_encoder_frozen else network.encoder
        head: EncoderHead = EncoderHead(pooling=network.pooling) if self.head is None else self.head
        return encoder, head
