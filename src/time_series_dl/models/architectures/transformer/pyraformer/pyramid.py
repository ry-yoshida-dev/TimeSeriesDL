import math
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class PyraformerPyramid:
    """
    Pyramidal graph of Pyraformer: the input steps form the finest scale, and each coarser scale has one node per
    ``window_sizes[scale]`` nodes of the scale below, the last node also taking any remainder.

    Nodes are numbered scale by scale, from the finest.

    Attributes
    ----------
    input_length : int
        Number of nodes of the finest scale.
    window_sizes : tuple[int, ...]
        Number of children of each node of every coarser scale, from the finest.
    inner_size : int
        Number of neighbours, centred on each node and including itself, that it attends to within its scale.

    Raises
    ------
    ValueError
        If a size is not positive, or a scale would have no nodes.
    """

    input_length: int
    window_sizes: tuple[int, ...]
    inner_size: int

    def __post_init__(self) -> None:
        if self.input_length <= 0 or self.inner_size <= 0:
            raise ValueError(
                f"input_length and inner_size must be positive, got {self.input_length} and {self.inner_size}"
            )
        if not self.window_sizes or any(size <= 0 for size in self.window_sizes):
            raise ValueError(f"window_sizes must be positive and not empty, got {self.window_sizes}")
        if self.input_length < math.prod(self.window_sizes):
            raise ValueError(
                f"input_length {self.input_length} must be at least the product of window_sizes {self.window_sizes}"
            )

    @property
    def scale_sizes(self) -> tuple[int, ...]:
        """
        Get the number of nodes of every scale.

        Returns
        -------
        tuple[int, ...]
            The sizes from the finest scale, each the previous one divided by its window size, rounded down.
        """
        sizes: list[int] = [self.input_length]
        for window_size in self.window_sizes:
            sizes.append(sizes[-1] // window_size)
        return tuple(sizes)

    @property
    def num_nodes(self) -> int:
        """
        Get the number of nodes of all scales.

        Returns
        -------
        int
            The sum of the scale sizes.
        """
        return sum(self.scale_sizes)

    def build_mask(self) -> Tensor:
        """
        Build the pyramidal attention mask: each node attends to its neighbours within its scale, to its parent and to
        its children.

        Returns
        -------
        Tensor
            Boolean mask with shape (num_nodes, num_nodes), True where the query node (row) may attend to the key
            node (column).
        """
        sizes: Tensor = torch.tensor(self.scale_sizes)
        scales: Tensor = torch.repeat_interleave(torch.arange(len(self.scale_sizes)), sizes)
        starts: Tensor = torch.tensor([0, *accumulate(self.scale_sizes)][:-1])
        local: Tensor = torch.arange(self.num_nodes) - starts[scales]
        is_same_scale: Tensor = scales.unsqueeze(1) == scales.unsqueeze(0)
        is_neighbour: Tensor = (local.unsqueeze(1) - local.unsqueeze(0)).abs() <= self.inner_size // 2
        parents: Tensor = self._parents(scales, local, sizes)
        is_parent: Tensor = (scales.unsqueeze(1) + 1 == scales.unsqueeze(0)) & (
            parents.unsqueeze(1) == local.unsqueeze(0)
        )
        return (is_same_scale & is_neighbour) | is_parent | is_parent.T

    def build_reference_indexes(self) -> Tensor:
        """
        Build the index of every input step's ancestor at each scale, whose features are concatenated per step.

        Returns
        -------
        Tensor
            Node indexes with shape (input_length, number of scales); column 0 is the step itself.
        """
        columns: list[Tensor] = [torch.arange(self.input_length)]
        start: int = 0
        local: Tensor = columns[0]
        for scale, window_size in enumerate(self.window_sizes):
            start += self.scale_sizes[scale]
            local = (local // window_size).clamp(max=self.scale_sizes[scale + 1] - 1)
            columns.append(start + local)
        return torch.stack(columns, dim=1)

    def _parents(self, scales: Tensor, local: Tensor, sizes: Tensor) -> Tensor:
        """
        Get the local index of each node's parent at the next coarser scale, or -1 at the coarsest scale.
        """
        windows: Tensor = torch.tensor([*self.window_sizes, 1])[scales]
        coarsest: int = len(self.window_sizes)
        next_sizes: Tensor = torch.cat([sizes[1:], sizes.new_ones(1)])[scales]
        parents: Tensor = torch.minimum(local // windows, next_sizes - 1)
        return torch.where(scales == coarsest, torch.full_like(parents, -1), parents)
