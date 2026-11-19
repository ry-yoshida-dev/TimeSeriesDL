from dataclasses import dataclass, field

from ....training import OptimizerKind, TrainingSettings


def _build_post_hoc_training() -> TrainingSettings:
    return TrainingSettings(
        max_epochs=20, batch_size=128, optimizer=OptimizerKind.ADAM, early_stopping_patience=None, random_seed=0
    )


@dataclass(frozen=True, kw_only=True)
class GenerationEvaluationSettings:
    """
    How synthetic sequences are compared with real ones.

    The post-hoc networks follow TimeGAN's protocol: a one-layer GRU with ``max(channels // 2, 1)`` hidden units,
    trained with Adam in batches of 128. The authors count iterations (2000 for the classifier, 5000 for the
    predictor); here the trainer counts epochs.

    Attributes
    ----------
    discriminator_training : TrainingSettings
        Training of the real-versus-synthetic classifier of the discriminative score.
    predictor_training : TrainingSettings
        Training of the next-step predictor of the predictive score.
    test_fraction : float
        Fraction of the real and of the synthetic sequences held out to measure the classifier's accuracy.
    split_seed : int
        Seed of the train-test split of the discriminative score.
    num_quantiles : int
        Number of quantiles of each channel compared by the marginal distance.
    max_lag : int | None
        Largest lag compared by the autocorrelation distance; every lag up to ``length - 1`` when None.

    Raises
    ------
    ValueError
        If test_fraction is not in (0, 1), or num_quantiles or max_lag is not positive.
    """

    discriminator_training: TrainingSettings = field(default_factory=_build_post_hoc_training)
    predictor_training: TrainingSettings = field(default_factory=_build_post_hoc_training)
    test_fraction: float = 0.2
    split_seed: int = 0
    num_quantiles: int = 100
    max_lag: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")
        if self.num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be positive, got {self.num_quantiles}")
        if self.max_lag is not None and self.max_lag <= 0:
            raise ValueError(f"max_lag must be positive when set, got {self.max_lag}")
