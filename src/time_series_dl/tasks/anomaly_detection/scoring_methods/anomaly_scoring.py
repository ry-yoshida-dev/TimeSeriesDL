from abc import ABC, abstractmethod

from ....losses import PointLoss
from ....models import ForecastingModel, ReconstructionModel, SequenceShape
from ....models.interfaces.protocols.anomaly_scoring import AnomalyScoringModel
from ..scoring_step import AnomalyScoringStep


class AnomalyScoring(ABC):
    """
    How a model is trained on normal windows and turned into an anomaly score of every time step.
    """

    @abstractmethod
    def is_supported(self, model: ReconstructionModel | ForecastingModel | AnomalyScoringModel) -> bool:
        """
        Check whether a model builds the network this scoring needs.

        Parameters
        ----------
        model : ReconstructionModel | ForecastingModel | AnomalyScoringModel
            The model parameters.

        Returns
        -------
        bool
            True if ``build_step`` accepts the model.
        """

    @abstractmethod
    def compute_sample_length(self, window_length: int) -> int:
        """
        Get the number of steps of each window the network is trained on and scores.

        Parameters
        ----------
        window_length : int
            The window length of the detector settings.

        Returns
        -------
        int
            The sample length.
        """

    @abstractmethod
    def build_step(
        self,
        model: ReconstructionModel | ForecastingModel | AnomalyScoringModel,
        shape: SequenceShape,
        loss: PointLoss,
    ) -> AnomalyScoringStep:
        """
        Build the untrained network of a model and its training and scoring step.

        Parameters
        ----------
        model : ReconstructionModel | ForecastingModel | AnomalyScoringModel
            The model parameters.
        shape : SequenceShape
            The window length of the detector settings and the number of channels.
        loss : PointLoss
            The point loss of reconstruction and forecast errors; unused by scorings that do not measure errors.

        Returns
        -------
        AnomalyScoringStep
            The step.

        Raises
        ------
        TypeError
            If the model does not build the network this scoring needs.
        """
