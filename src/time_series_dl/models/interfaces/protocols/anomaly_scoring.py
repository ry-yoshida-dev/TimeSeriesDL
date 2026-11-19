"""Structural type for models that score anomalies with their own objective."""

from typing import Protocol, runtime_checkable

from ..anomaly_scoring_network import AnomalyScoringNetwork
from ..shapes import SequenceShape


@runtime_checkable
class AnomalyScoringModel(Protocol):
    """
    Model parameters able to build an anomaly scoring network, for anomaly detection with the model's own training
    objective and anomaly score.
    """

    def build_anomaly_scoring_network(self, shape: SequenceShape) -> AnomalyScoringNetwork:
        """
        Build an untrained anomaly scoring network.

        Parameters
        ----------
        shape : SequenceShape
            The dimensions of the windows it scores.

        Returns
        -------
        AnomalyScoringNetwork
            The network.
        """
        ...
