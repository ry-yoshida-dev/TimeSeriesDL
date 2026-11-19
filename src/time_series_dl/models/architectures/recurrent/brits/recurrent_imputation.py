from typing import cast

import torch
from torch import Tensor, nn

from ....components.missing_values import TemporalDecay, TimeGap
from .feature_regression import BRITSFeatureRegression


class BRITSRecurrentImputation(nn.Module):
    """
    RITS, the recurrent imputation of one direction of BRITS (Cao et al., 2018).

    At each step the decayed hidden state gives a history-based estimate, the feature regression gives an estimate
    from the other channels, and the two are combined by weights learned from the input decay and the mask. The
    combined estimate fills the missing inputs of an LSTM cell.

    Parameters
    ----------
    channels : int
        Number of channels.
    hidden_size : int
        Size of the hidden state.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, channels: int, hidden_size: int) -> None:
        if channels <= 0 or hidden_size <= 0:
            raise ValueError(f"channels and hidden_size must be positive, got {channels} and {hidden_size}")
        super().__init__()
        self.hidden_size: int = hidden_size
        self.hidden_decay: TemporalDecay = TemporalDecay(channels, hidden_size, is_diagonal=False)
        self.input_decay: TemporalDecay = TemporalDecay(channels, channels, is_diagonal=True)
        self.history_regression: nn.Linear = nn.Linear(hidden_size, channels)
        self.feature_regression: BRITSFeatureRegression = BRITSFeatureRegression(channels)
        self.combination: nn.Linear = nn.Linear(2 * channels, channels)
        self.cell: nn.LSTMCell = nn.LSTMCell(2 * channels, hidden_size)

    def forward(self, values: Tensor, is_observed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Impute a batch forward in time.

        Parameters
        ----------
        values : Tensor
            Values with shape (batch, length, channels); missing values are zero.
        is_observed : Tensor
            Boolean mask of the shape of ``values``.

        Returns
        -------
        tuple[Tensor, Tensor]
            The combined estimates of every value, of the shape of ``values``, and the scalar estimation loss: the
            mean absolute error of the history, feature and combined estimates on the observed values, averaged over
            the steps.
        """
        mask: Tensor = is_observed.to(values.dtype)
        gaps: Tensor = TimeGap.compute(is_observed)
        hidden_decays: Tensor = self.hidden_decay(gaps)
        combining_weights: Tensor = torch.sigmoid(self.combination(torch.cat([self.input_decay(gaps), mask], dim=2)))
        hidden: Tensor = values.new_zeros(values.shape[0], self.hidden_size)
        cell_state: Tensor = values.new_zeros(values.shape[0], self.hidden_size)
        history_estimates: list[Tensor] = []
        feature_estimates: list[Tensor] = []
        combined_estimates: list[Tensor] = []
        for step in range(values.shape[1]):
            observed: Tensor = values[:, step]
            step_mask: Tensor = mask[:, step]
            hidden = hidden * hidden_decays[:, step]
            history: Tensor = self.history_regression(hidden)
            feature: Tensor = self.feature_regression(step_mask * observed + (1.0 - step_mask) * history)
            weight: Tensor = combining_weights[:, step]
            combined: Tensor = weight * feature + (1.0 - weight) * history
            history_estimates.append(history)
            feature_estimates.append(feature)
            combined_estimates.append(combined)
            complemented: Tensor = step_mask * observed + (1.0 - step_mask) * combined
            hidden, cell_state = self.cell(torch.cat([complemented, step_mask], dim=1), (hidden, cell_state))
        estimates: Tensor = torch.stack(combined_estimates, dim=1)
        loss: Tensor = (
            self._step_error(torch.stack(history_estimates, dim=1), values, mask)
            + self._step_error(torch.stack(feature_estimates, dim=1), values, mask)
            + self._step_error(estimates, values, mask)
        ) / 3.0
        return estimates, loss

    @staticmethod
    def _step_error(estimates: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        # Mean absolute error over the observed values of each step, averaged over the steps.
        errors: Tensor = ((estimates - values).abs() * mask).sum(dim=(0, 2))
        return (errors / mask.sum(dim=(0, 2)).clamp(min=1.0)).mean()

    def __call__(self, values: Tensor, is_observed: Tensor) -> tuple[Tensor, Tensor]:
        return cast(tuple[Tensor, Tensor], super().__call__(values, is_observed))
