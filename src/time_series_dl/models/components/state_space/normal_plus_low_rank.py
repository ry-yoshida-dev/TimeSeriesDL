from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from .hippo_measure import HiPPOMeasure


@dataclass(frozen=True)
class NormalPlusLowRank:
    """
    The HiPPO-LegS state matrix written as a normal matrix minus a rank-one term, A = V Lambda V* - P P^T, the
    starting point of S4, S4D-LegS and S5 (Gu et al., 2022).

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues Lambda of the normal part, with shape (state_size,).
    low_rank : Tensor
        Complex low-rank vector V* P in the eigenbasis, with shape (state_size,).
    input_vector : Tensor
        Complex input vector V* B in the eigenbasis, with shape (state_size,).
    eigenvectors : Tensor
        Complex unitary eigenvectors V with shape (state_size, state_size).
    """

    eigenvalues: Tensor
    low_rank: Tensor
    input_vector: Tensor
    eigenvectors: Tensor

    @staticmethod
    def from_legs(state_size: int) -> NormalPlusLowRank:
        """
        Diagonalize the normal part of HiPPO-LegS, in complex128.

        Eigenvalues are sorted so that conjugate pairs fall in different halves, with positive imaginary parts in the
        second half.

        Parameters
        ----------
        state_size : int
            Size of the state matrix.

        Returns
        -------
        NormalPlusLowRank
            The decomposition.
        """
        transition, input_matrix = HiPPOMeasure.LEGS.transition(state_size)
        low_rank: Tensor = torch.sqrt(torch.arange(state_size, dtype=torch.float64) + 0.5)
        normal: Tensor = transition + low_rank[:, None] * low_rank[None, :]
        # The normal part is -1/2 I plus a skew-symmetric matrix, so -i times it is Hermitian up to the real shift.
        real_part: Tensor = torch.diagonal(normal).mean()
        decomposition: tuple[Tensor, Tensor] = cast(
            tuple[Tensor, Tensor], torch.linalg.eigh(normal.to(torch.complex128) * -1j)
        )
        imaginary_parts: Tensor = decomposition[0]
        eigenvectors: Tensor = decomposition[1]
        adjoint: Tensor = eigenvectors.conj().T
        return NormalPlusLowRank(
            eigenvalues=torch.complex(real_part.expand_as(imaginary_parts), imaginary_parts),
            low_rank=adjoint @ low_rank.to(torch.complex128),
            input_vector=adjoint @ input_matrix.squeeze(-1).to(torch.complex128),
            eigenvectors=eigenvectors,
        )
