from typing import cast

from torch import Tensor, nn


class DiffusionTSAdaptiveLayerNorm(nn.Module):
    """
    Layer normalization without learned affine parameters, followed by a scale and shift computed from the diffusion
    step embedding: ``norm(x) * (1 + scale) + shift``.

    Parameters
    ----------
    model_size : int
        Size of each token vector, also of the step embedding.
    """

    def __init__(self, model_size: int) -> None:
        super().__init__()
        self.normalization: nn.LayerNorm = nn.LayerNorm(model_size, elementwise_affine=False)
        self.modulation: nn.Linear = nn.Linear(model_size, 2 * model_size)

    def forward(self, tokens: Tensor, step_embedding: Tensor) -> Tensor:
        """
        Normalize tokens and modulate them by the diffusion step.

        Parameters
        ----------
        tokens : Tensor
            Tokens with shape (batch, length, model_size).
        step_embedding : Tensor
            Embedding of each sample's diffusion step with shape (batch, model_size).

        Returns
        -------
        Tensor
            The modulated tokens with the shape of the input.
        """
        scale: Tensor
        shift: Tensor
        scale, shift = self.modulation(step_embedding).unsqueeze(1).chunk(2, dim=2)
        modulated: Tensor = self.normalization(tokens) * (1.0 + scale) + shift
        return modulated

    def __call__(self, tokens: Tensor, step_embedding: Tensor) -> Tensor:
        return cast(Tensor, super().__call__(tokens, step_embedding))
