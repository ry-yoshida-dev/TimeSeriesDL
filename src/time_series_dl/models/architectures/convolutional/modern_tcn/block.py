from typing import cast

from torch import Tensor, nn

from .large_kernel_convolution import ModernTCNLargeKernelConvolution
from .stage import ModernTCNStage


class ModernTCNBlock(nn.Module):
    """
    ModernTCN block: a large-kernel depthwise convolution along time (DWConv), batch normalization, a pointwise
    feed-forward network over the features of each variable (ConvFFN1) and one over the variables of each feature
    (ConvFFN2), with a residual connection around the whole block.

    Parameters
    ----------
    stage : ModernTCNStage
        The sizes of the stage the block belongs to.
    num_variables : int
        Number of variables (input channels).
    feedforward_ratio : int
        Ratio of the hidden size of each feed-forward network to its input size.
    dropout : float
        Dropout probability inside the feed-forward networks.
    """

    def __init__(self, stage: ModernTCNStage, num_variables: int, feedforward_ratio: int, dropout: float) -> None:
        super().__init__()
        model_size: int = stage.model_size
        channels: int = num_variables * model_size
        hidden_channels: int = channels * feedforward_ratio
        self.depthwise: ModernTCNLargeKernelConvolution = ModernTCNLargeKernelConvolution(
            channels, stage.large_kernel_size, stage.small_kernel_size
        )
        self.normalization: nn.BatchNorm1d = nn.BatchNorm1d(model_size)
        self.variable_feedforward: nn.Sequential = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, groups=num_variables),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, 1, groups=num_variables),
            nn.Dropout(dropout),
        )
        self.feature_feedforward: nn.Sequential = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, groups=model_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Conv1d(hidden_channels, channels, 1, groups=model_size),
            nn.Dropout(dropout),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Apply the block.

        Parameters
        ----------
        features : Tensor
            Features with shape (batch, num_variables, model_size, num_patches).

        Returns
        -------
        Tensor
            Features with the shape of ``features``.
        """
        batch_size: int
        num_variables: int
        model_size: int
        num_patches: int
        batch_size, num_variables, model_size, num_patches = features.shape
        mixed: Tensor = self.depthwise(features.reshape(batch_size, num_variables * model_size, num_patches))
        normalized: Tensor = self.normalization(mixed.reshape(batch_size * num_variables, model_size, num_patches))
        per_variable: Tensor = self.variable_feedforward(
            normalized.reshape(batch_size, num_variables * model_size, num_patches)
        )
        by_feature: Tensor = per_variable.reshape(batch_size, num_variables, model_size, num_patches).transpose(1, 2)
        per_feature: Tensor = self.feature_feedforward(by_feature.reshape(batch_size, model_size * num_variables, -1))
        output: Tensor = per_feature.reshape(batch_size, model_size, num_variables, num_patches).transpose(1, 2)
        return features + output

    def __call__(self, features: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(features))
