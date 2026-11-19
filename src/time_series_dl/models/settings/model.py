from collections.abc import Mapping
from dataclasses import dataclass, field

from ..interfaces import (
    AnomalyScoringModel,
    ForecastingModel,
    IrregularInterpolationModel,
    IrregularSummaryModel,
    MaskedReconstructionModel,
    PretrainingModel,
    ReconstructionModel,
    SequenceGenerationModel,
    SummaryModel,
)
from .architecture import ModelArchitecture, ModelParameters
from .parameters_reader import ParametersReader


@dataclass(frozen=True, kw_only=True)
class ModelSettings:
    """
    Architecture and parameters of a model, in a form a configuration file can describe.

    The fields map one-to-one onto the ``model`` section of the YAML presets in this package, so a preset can be
    turned into settings by any dataclass builder (enums are written by value). The parameters are checked when the
    settings are created.

    Attributes
    ----------
    architecture : ModelArchitecture
        Model family.
    parameters : Mapping[str, object]
        Fields of the parameters class of ``architecture``; fields left out keep their defaults.

    Raises
    ------
    TypeError
        If a parameter has the wrong type.
    KeyError
        If a parameter is not a field of the parameters class.
    ValueError
        If a parameter is not a member of its enum, or the parameters class rejects a value.
    """

    architecture: ModelArchitecture
    parameters: Mapping[str, object] = field(default_factory=dict[str, object])

    def __post_init__(self) -> None:
        self.build()

    def build(self) -> ModelParameters:
        """
        Build the parameters of ``architecture``.

        Returns
        -------
        ModelParameters
            The parameters, such as ``PatchTSTParameters``.
        """
        return ParametersReader.read(self.architecture.parameters_type, self.parameters)

    def build_forecasting_model(self) -> ForecastingModel:
        """
        Build the parameters for ``ForecasterSettings``.

        Returns
        -------
        ForecastingModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not support forecasting.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, ForecastingModel):
            raise ValueError(f"{self.architecture} does not support forecasting.")
        return parameters

    def build_summary_model(self) -> SummaryModel:
        """
        Build the parameters for ``ClassifierSettings`` or ``RegressorSettings``.

        Returns
        -------
        SummaryModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not support classification and regression.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, SummaryModel):
            raise ValueError(f"{self.architecture} does not support classification or regression.")
        return parameters

    def build_reconstruction_model(self) -> ReconstructionModel:
        """
        Build the parameters of a reconstruction model, for ``AnomalyDetectorSettings`` or ``ImputerSettings``.

        Returns
        -------
        ReconstructionModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not support imputation and anomaly detection.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, ReconstructionModel):
            raise ValueError(f"{self.architecture} does not support imputation or anomaly detection.")
        return parameters

    def build_imputation_model(self) -> ReconstructionModel | MaskedReconstructionModel:
        """
        Build the parameters for ``ImputerSettings``, preferring a mask-aware network when the model has one.

        Returns
        -------
        ReconstructionModel | MaskedReconstructionModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not support imputation.
        """
        parameters: ModelParameters = self.build()
        if isinstance(parameters, MaskedReconstructionModel):
            return parameters
        if isinstance(parameters, ReconstructionModel):
            return parameters
        raise ValueError(f"{self.architecture} does not support imputation.")

    def build_pretraining_model(self) -> PretrainingModel:
        """
        Build the parameters for ``RepresentationLearnerSettings``.

        Returns
        -------
        PretrainingModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` has no self-supervised pretraining method.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, PretrainingModel):
            raise ValueError(f"{self.architecture} does not support self-supervised pretraining.")
        return parameters

    def build_anomaly_detection_model(self) -> ReconstructionModel | ForecastingModel | AnomalyScoringModel:
        """
        Build the parameters for ``AnomalyDetectorSettings``, preferring the model's own anomaly score.

        Returns
        -------
        ReconstructionModel | ForecastingModel | AnomalyScoringModel
            The parameters; the detector scores by the model's own score, the reconstruction error or the forecast
            error, in that order of preference.

        Raises
        ------
        ValueError
            If ``architecture`` supports no anomaly scoring.
        """
        parameters: ModelParameters = self.build()
        if isinstance(parameters, AnomalyScoringModel):
            return parameters
        if isinstance(parameters, ReconstructionModel):
            return parameters
        if isinstance(parameters, ForecastingModel):
            return parameters
        raise ValueError(f"{self.architecture} does not support anomaly detection.")

    def build_irregular_summary_model(self) -> IrregularSummaryModel:
        """
        Build the parameters for ``IrregularClassifierSettings`` or ``IrregularRegressorSettings``.

        Returns
        -------
        IrregularSummaryModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not classify or regress irregularly sampled sequences.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, IrregularSummaryModel):
            raise ValueError(f"{self.architecture} does not support irregular classification or regression.")
        return parameters

    def build_irregular_interpolation_model(self) -> IrregularInterpolationModel:
        """
        Build the parameters for ``IrregularInterpolatorSettings``.

        Returns
        -------
        IrregularInterpolationModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not interpolate irregularly sampled sequences.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, IrregularInterpolationModel):
            raise ValueError(f"{self.architecture} does not support irregular interpolation.")
        return parameters

    def build_sequence_generation_model(self) -> SequenceGenerationModel:
        """
        Build the parameters for ``GeneratorSettings``.

        Returns
        -------
        SequenceGenerationModel
            The parameters.

        Raises
        ------
        ValueError
            If ``architecture`` does not generate sequences.
        """
        parameters: ModelParameters = self.build()
        if not isinstance(parameters, SequenceGenerationModel):
            raise ValueError(f"{self.architecture} does not support generation.")
        return parameters
