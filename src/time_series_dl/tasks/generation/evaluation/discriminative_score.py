from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from ....array_types import FloatArray, IndexArray
from ....runtime import BatchedInference
from ....training import Trainer, TrainingSettings
from .discrimination_step import DiscriminationStep
from .recurrent_scorer import RecurrentScorer


@dataclass(frozen=True)
class DiscriminativeScore:
    """
    Discriminative score of TimeGAN (Yoon et al., 2019): a post-hoc GRU classifier is trained to tell real from
    synthetic sequences, and the score is ``|accuracy - 0.5|`` on held-out sequences; 0 means the classifier cannot
    tell them apart, 0.5 that it always can.

    Attributes
    ----------
    training : TrainingSettings
        Training of the classifier.
    test_fraction : float
        Fraction of the real and of the synthetic sequences held out for the accuracy.
    split_seed : int
        Seed of the train-test split.
    """

    training: TrainingSettings
    test_fraction: float
    split_seed: int

    def compute(self, real: FloatArray, synthetic: FloatArray) -> float:
        """
        Compute the score of synthetic sequences.

        Parameters
        ----------
        real : FloatArray
            Real sequences with shape (num_real, length, channels), at least two.
        synthetic : FloatArray
            Synthetic sequences with shape (num_synthetic, length, channels), at least two.

        Returns
        -------
        float
            The score in [0, 0.5]; lower is better.

        Raises
        ------
        ValueError
            If there are fewer than two sequences of a kind or their shapes differ.
        """
        if real.shape[1:] != synthetic.shape[1:]:
            raise ValueError(f"real and synthetic sequences differ in shape: {real.shape} and {synthetic.shape}")
        if real.shape[0] < 2 or synthetic.shape[0] < 2:
            raise ValueError("the discriminative score needs at least two real and two synthetic sequences.")
        random: np.random.Generator = np.random.default_rng(self.split_seed)
        real_test: IndexArray
        real_train: IndexArray
        synthetic_test: IndexArray
        synthetic_train: IndexArray
        real_test, real_train = self._split(real.shape[0], random)
        synthetic_test, synthetic_train = self._split(synthetic.shape[0], random)
        training_data: TensorDataset = self._build_dataset(real[real_train], synthetic[synthetic_train])
        test_data: TensorDataset = self._build_dataset(real[real_test], synthetic[synthetic_test])
        trainer: Trainer = Trainer(self.training)
        trainer.seed()
        channels: int = int(real.shape[2])
        network: RecurrentScorer = RecurrentScorer(channels, max(channels // 2, 1), 1)
        _ = trainer.fit(network, DiscriminationStep(network), training_data)
        logits: Tensor = BatchedInference(trainer.device, self.training.batch_size).run(
            lambda batch: network(batch[0])[:, -1, 0], TensorDataset(test_data.tensors[0])
        )
        is_correct: Tensor = (logits > 0.0) == test_data.tensors[1].bool()
        return abs(float(is_correct.double().mean()) - 0.5)

    def _split(self, count: int, random: np.random.Generator) -> tuple[IndexArray, IndexArray]:
        order: IndexArray = random.permutation(count)
        num_test: int = min(max(round(self.test_fraction * count), 1), count - 1)
        return order[:num_test], order[num_test:]

    @staticmethod
    def _build_dataset(real: FloatArray, synthetic: FloatArray) -> TensorDataset:
        values: FloatArray = np.concatenate([real, synthetic], axis=0).astype(np.float32)
        labels: FloatArray = np.concatenate([np.ones(real.shape[0]), np.zeros(synthetic.shape[0])]).astype(np.float32)
        return TensorDataset(torch.from_numpy(values), torch.from_numpy(labels))
