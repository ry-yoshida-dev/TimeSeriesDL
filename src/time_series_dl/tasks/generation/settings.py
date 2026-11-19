from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_scaler import MinMaxScalingParameters, ScalingParameters

from ...models.interfaces.protocols.sequence_generation import SequenceGenerationModel
from ...training import TrainingSettings
from .evaluation import GenerationEvaluationSettings

if TYPE_CHECKING:
    from .generator import SequenceGenerator


@dataclass(frozen=True, kw_only=True)
class GeneratorSettings:
    """
    Everything needed to train a generator of synthetic sequences, identical for every model.

    Attributes
    ----------
    model : SequenceGenerationModel
        Parameters of the generative model, such as ``TimeGANParameters`` or ``DiffusionTSParameters``.
    training : TrainingSettings
        Optimizer, epochs, early stopping and device; the model's loss decides what is monitored.
    scaling : ScalingParameters
        Per-channel scaling fitted on the training sequences; it must be invertible, so samples return to the
        original scale. Min-max scaling to [0, 1] by default, as TimeGAN and Diffusion-TS expect.
    window_length : int | None
        Length of the windows ``fit_series`` cuts from long series; None when only fitting on sequences.
    window_stride : int
        Steps between consecutive windows of ``fit_series``.
    evaluation : GenerationEvaluationSettings
        How ``evaluate`` compares synthetic sequences with real ones.

    Raises
    ------
    ValueError
        If window_length or window_stride is not positive.
    """

    model: SequenceGenerationModel
    training: TrainingSettings = field(default_factory=TrainingSettings)
    scaling: ScalingParameters = field(default_factory=MinMaxScalingParameters)
    window_length: int | None = None
    window_stride: int = 1
    evaluation: GenerationEvaluationSettings = field(default_factory=GenerationEvaluationSettings)

    def __post_init__(self) -> None:
        if self.window_length is not None and self.window_length <= 0:
            raise ValueError(f"window_length must be positive when set, got {self.window_length}")
        if self.window_stride <= 0:
            raise ValueError(f"window_stride must be positive, got {self.window_stride}")

    def build(self) -> SequenceGenerator:
        """
        Build an unfitted generator.

        Returns
        -------
        SequenceGenerator
            The generator.

        Raises
        ------
        ValueError
            If the scaling strategy mixes channels or is not invertible.
        """
        from .generator import SequenceGenerator

        return SequenceGenerator(self)
