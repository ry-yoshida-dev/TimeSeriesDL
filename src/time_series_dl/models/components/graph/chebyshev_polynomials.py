from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, kw_only=True)
class ChebyshevPolynomials:
    """
    Chebyshev polynomials of a square matrix, ``T_0 = I``, ``T_1 = M`` and ``T_k = 2 M T_(k-1) - T_(k-2)``, which
    spectral graph convolutions use as a polynomial basis of graph filters.

    Attributes
    ----------
    num_terms : int
        Number of polynomials, from degree 0 to num_terms - 1.

    Raises
    ------
    ValueError
        If num_terms is not positive.
    """

    num_terms: int

    def __post_init__(self) -> None:
        if self.num_terms <= 0:
            raise ValueError(f"num_terms must be positive, got {self.num_terms}")

    def evaluate(self, matrix: Tensor) -> Tensor:
        """
        Evaluate the polynomials of a matrix or a batch of matrices.

        Parameters
        ----------
        matrix : Tensor
            Matrices with shape (..., nodes, nodes).

        Returns
        -------
        Tensor
            Polynomials with shape (..., num_terms, nodes, nodes).

        Raises
        ------
        ValueError
            If the last two dimensions are not equal.
        """
        if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must be square in its last two dimensions, got shape {tuple(matrix.shape)}")
        identity: Tensor = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        terms: list[Tensor] = [identity, matrix][: self.num_terms]
        while len(terms) < self.num_terms:
            terms.append(2.0 * matrix @ terms[-1] - terms[-2])
        return torch.stack(terms, dim=-3)
