from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class BatchedInference:
    """
    Runs a function of tensor batches over a whole dataset without gradients.

    Attributes
    ----------
    device : torch.device
        Device the batches are moved to.
    batch_size : int
        Number of samples per batch.
    """

    device: torch.device
    batch_size: int

    def run(self, function: Callable[[tuple[Tensor, ...]], Tensor], data: TensorDataset) -> Tensor:
        """
        Apply a function to every batch and concatenate the results on the CPU.

        Parameters
        ----------
        function : Callable[[tuple[Tensor, ...]], Tensor]
            Maps the tensors of one batch to an output whose first dimension is the batch.
        data : TensorDataset
            The samples.

        Returns
        -------
        Tensor
            The concatenated outputs.

        Raises
        ------
        ValueError
            If the dataset is empty.
        """
        if len(data) == 0:
            raise ValueError("data must not be empty.")
        loader: DataLoader[tuple[Tensor, ...]] = DataLoader(data, batch_size=self.batch_size)
        outputs: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                moved: tuple[Tensor, ...] = tuple(tensor.to(self.device) for tensor in batch)
                outputs.append(function(moved).cpu())
        return torch.cat(outputs, dim=0)
