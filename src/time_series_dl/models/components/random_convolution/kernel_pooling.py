from collections.abc import Sequence
from enum import Enum

import torch
from torch import Tensor


class KernelPooling(Enum):
    """
    Reduction of the convolution output of one kernel and bias over time to one feature, as in the ROCKET family.

    Every pooling reads the differences between the convolution output and the bias, and only the time steps marked
    valid, so kernels without padding ignore the steps where the kernel overlaps the padding.

    Attributes
    ----------
    PROPORTION_OF_POSITIVE_VALUES
        Fraction of valid steps whose difference is positive (PPV).
    MAX
        Largest difference (the max feature of ROCKET).
    MEAN_OF_POSITIVE_VALUES
        Mean of the positive differences, zero when there is none (MPV of MultiRocket).
    MEAN_OF_INDICES_OF_POSITIVE_VALUES
        Mean index, counted from the first valid step, of the positive differences, minus one when there is none
        (MIPV of MultiRocket).
    LONGEST_STRETCH_OF_POSITIVE_VALUES
        Length of the longest run of consecutive positive differences (LSPV of MultiRocket).
    """

    PROPORTION_OF_POSITIVE_VALUES = "proportion_of_positive_values"
    MAX = "max"
    MEAN_OF_POSITIVE_VALUES = "mean_of_positive_values"
    MEAN_OF_INDICES_OF_POSITIVE_VALUES = "mean_of_indices_of_positive_values"
    LONGEST_STRETCH_OF_POSITIVE_VALUES = "longest_stretch_of_positive_values"

    @staticmethod
    def pool(poolings: Sequence["KernelPooling"], differences: Tensor, is_valid: Tensor) -> Tensor:
        """
        Pool the differences of a batch of convolution outputs and biases by several poolings at once.

        Parameters
        ----------
        poolings : Sequence[KernelPooling]
            The poolings, in output order.
        differences : Tensor
            Convolution outputs minus biases with shape (batch, kernels, biases, length).
        is_valid : Tensor
            Boolean mask of the steps each kernel pools, with shape (kernels, length); every row has a valid step.

        Returns
        -------
        Tensor
            Features with shape (batch, kernels, biases, len(poolings)).
        """
        is_valid_step: Tensor = is_valid[None, :, None, :]
        is_positive: Tensor = (differences > 0) & is_valid_step
        count: Tensor = is_positive.sum(dim=-1).to(differences.dtype)
        features: list[Tensor] = [
            pooling._reduce(differences, is_valid_step, is_positive, count) for pooling in poolings
        ]
        return torch.stack(features, dim=-1)

    def _reduce(self, differences: Tensor, is_valid_step: Tensor, is_positive: Tensor, count: Tensor) -> Tensor:
        match self:
            case KernelPooling.PROPORTION_OF_POSITIVE_VALUES:
                return count / is_valid_step.sum(dim=-1).to(differences.dtype)
            case KernelPooling.MAX:
                return differences.masked_fill(~is_valid_step, float("-inf")).amax(dim=-1)
            case KernelPooling.MEAN_OF_POSITIVE_VALUES:
                return (differences * is_positive).sum(dim=-1) / count.clamp(min=1.0)
            case KernelPooling.MEAN_OF_INDICES_OF_POSITIVE_VALUES:
                positions: Tensor = torch.arange(differences.shape[-1], device=differences.device)
                first_valid: Tensor = is_valid_step.to(torch.int64).argmax(dim=-1, keepdim=True)
                indices: Tensor = (positions - first_valid).to(differences.dtype)
                index_sum: Tensor = (indices * is_positive).sum(dim=-1)
                return torch.where(count > 0, index_sum / count.clamp(min=1.0), -1.0)
            case KernelPooling.LONGEST_STRETCH_OF_POSITIVE_VALUES:
                steps: Tensor = torch.arange(differences.shape[-1], device=differences.device, dtype=torch.int32)
                last_break: Tensor = torch.where(is_positive, -1, steps).cummax(dim=-1).values
                run_lengths: Tensor = torch.where(is_positive, steps - last_break, 0)
                return run_lengths.amax(dim=-1).to(differences.dtype)
