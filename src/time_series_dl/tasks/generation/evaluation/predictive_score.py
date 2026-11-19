from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ....array_types import FloatArray
from ....runtime import BatchedInference
from ....training import Trainer, TrainingSettings
from .next_step_prediction_step import NextStepPredictionStep
from .recurrent_scorer import RecurrentScorer


@dataclass(frozen=True)
class PredictiveScore:
    """
    Predictive score of TimeGAN (Yoon et al., 2019), "train on synthetic, test on real": a post-hoc GRU learns on the
    synthetic sequences to predict the last channel one step ahead from the other channels, and the score is its mean
    absolute error on the real sequences. A univariate sequence predicts its next value from its current one.

    Attributes
    ----------
    training : TrainingSettings
        Training of the predictor.
    """

    training: TrainingSettings

    def compute(self, real: FloatArray, synthetic: FloatArray) -> float:
        """
        Compute the score of synthetic sequences.

        Parameters
        ----------
        real : FloatArray
            Real sequences with shape (num_real, length, channels), at least two steps long.
        synthetic : FloatArray
            Synthetic sequences with the same length and channels.

        Returns
        -------
        float
            The mean absolute error on the real sequences; lower is better.

        Raises
        ------
        ValueError
            If the shapes differ or the sequences are shorter than two steps.
        """
        if real.shape[1:] != synthetic.shape[1:]:
            raise ValueError(f"real and synthetic sequences differ in shape: {real.shape} and {synthetic.shape}")
        if real.shape[1] < 2:
            raise ValueError(f"the predictive score needs sequences of at least 2 steps, got {real.shape[1]}")
        channels: int = int(real.shape[2])
        num_inputs: int = max(channels - 1, 1)
        trainer: Trainer = Trainer(self.training)
        trainer.seed()
        network: RecurrentScorer = RecurrentScorer(num_inputs, max(channels // 2, 1), 1)
        _ = trainer.fit(network, NextStepPredictionStep(network), self._build_dataset(synthetic, num_inputs))
        test_data: TensorDataset = self._build_dataset(real, num_inputs)
        predictions: Tensor = BatchedInference(trainer.device, self.training.batch_size).run(
            lambda batch: network(batch[0]), TensorDataset(test_data.tensors[0])
        )
        return float((predictions - test_data.tensors[1]).abs().double().mean())

    @staticmethod
    def _build_dataset(values: FloatArray, num_inputs: int) -> TensorDataset:
        sequences: FloatArray = np.asarray(values, dtype=np.float32)
        return TensorDataset(
            torch.from_numpy(np.ascontiguousarray(sequences[:, :-1, :num_inputs])),
            torch.from_numpy(np.ascontiguousarray(sequences[:, 1:, -1:])),
        )
