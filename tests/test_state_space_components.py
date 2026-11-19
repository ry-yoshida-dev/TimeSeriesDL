from typing import cast

import numpy as np
import pytest
import torch
from scipy.special import eval_legendre
from torch import Tensor, nn

from time_series_dl.models import DiagonalInitialization, Discretization, HiPPOMeasure
from time_series_dl.models.components.state_space import (
    CausalDepthwiseConvolution,
    DiagonalSSMKernel,
    DPLRSSMKernel,
    FFTConvolution,
    HiPPOMemory,
    LinearRecurrence,
    Mamba2StateSpaceLayer,
    NormalPlusLowRank,
)


def sequential_recurrence(decays: Tensor, inputs: Tensor) -> Tensor:
    state: Tensor = torch.zeros_like(inputs[:, 0])
    states: list[Tensor] = []
    for index in range(inputs.shape[1]):
        state = decays[:, index] * state + inputs[:, index]
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("length", [1, 2, 7, 16])
def test_scan_matches_sequential_recurrence(length: int) -> None:
    _ = torch.manual_seed(0)
    decays: Tensor = torch.rand(2, length, 3)
    inputs: Tensor = torch.randn(2, length, 3)
    assert torch.allclose(LinearRecurrence.scan(decays, inputs), sequential_recurrence(decays, inputs), atol=1e-6)
    complex_decays: Tensor = torch.polar(torch.rand(2, length, 3), torch.randn(2, length, 3))
    complex_inputs: Tensor = torch.randn(2, length, 3, dtype=torch.complex64)
    assert torch.allclose(
        LinearRecurrence.scan(complex_decays, complex_inputs),
        sequential_recurrence(complex_decays, complex_inputs),
        atol=1e-5,
    )


def test_scan_broadcasts_decays() -> None:
    decays: Tensor = torch.full((1, 1, 2), 0.5)
    inputs: Tensor = torch.ones(1, 3, 2)
    expected: Tensor = torch.tensor([1.0, 1.5, 1.75])[None, :, None].expand(1, 3, 2)
    assert torch.allclose(LinearRecurrence.scan(decays, inputs), expected)


def test_fft_convolution_is_causal_convolution() -> None:
    _ = torch.manual_seed(0)
    values: Tensor = torch.randn(2, 9)
    kernel: Tensor = torch.randn(2, 9)
    expected: Tensor = torch.stack(
        [(kernel[:, : step + 1].flip(1) * values[:, : step + 1]).sum(dim=1) for step in range(9)], dim=1
    )
    assert torch.allclose(FFTConvolution.apply(values, kernel), expected, atol=1e-5)


def test_causal_depthwise_convolution_matches_grouped_convolution() -> None:
    _ = torch.manual_seed(0)
    convolution: CausalDepthwiseConvolution = CausalDepthwiseConvolution(3, 4)
    reference: nn.Conv1d = nn.Conv1d(3, 3, 4, groups=3, padding=3)
    with torch.no_grad():
        _ = reference.weight.copy_(convolution.weight[:, None, :])
        assert reference.bias is not None
        _ = reference.bias.copy_(convolution.bias)
        values: Tensor = torch.randn(2, 10, 3)
        expected: Tensor = reference(values.transpose(1, 2))[..., :10].transpose(1, 2)
        assert torch.allclose(convolution(values), expected, atol=1e-6)


@pytest.mark.parametrize("discretization", list(Discretization))
def test_legendre_memory_reconstructs_its_window(discretization: Discretization) -> None:
    window_length: int = 64
    state_size: int = 32
    memory: HiPPOMemory = HiPPOMemory(HiPPOMeasure.LEGT, state_size, 200, float(window_length), discretization)
    steps: Tensor = torch.arange(200, dtype=torch.float32)
    signal: Tensor = torch.sin(steps / 7.0) + steps / 100.0
    state: Tensor = torch.zeros(state_size)
    for index in range(200):
        state = memory.step(state, signal[index], index)
    positions: np.ndarray = (np.arange(window_length) + 0.5) / window_length
    evaluation: Tensor = torch.tensor(
        eval_legendre(np.arange(state_size)[None, :], 1.0 - 2.0 * positions[:, None]), dtype=torch.float32
    )
    assert torch.allclose(evaluation @ state, signal[-window_length:], atol=0.05)
    response: Tensor = memory.impulse_response(200)
    assert torch.allclose((response.flip(0) * signal[:, None]).sum(dim=0), state, atol=1e-4)


def test_scaled_legendre_memory_reconstructs_whole_history() -> None:
    state_size: int = 32
    length: int = 100
    memory: HiPPOMemory = HiPPOMemory(HiPPOMeasure.LEGS, state_size, length, 1.0, Discretization.BILINEAR)
    signal: Tensor = torch.cos(torch.arange(length, dtype=torch.float32) / 10.0)
    state: Tensor = torch.zeros(state_size)
    for index in range(length):
        state = memory.step(state, signal[index], index)
    positions: np.ndarray = (np.arange(length) + 0.5) / length
    orders: np.ndarray = np.arange(state_size)
    # The normalized LegS state reconstructs sum_n c_n sqrt(2n + 1) P_n(2r - 1) over the whole history.
    evaluation: np.ndarray = eval_legendre(orders[None, :], 2.0 * positions[:, None] - 1.0) * np.sqrt(2 * orders + 1)
    reconstructed: Tensor = torch.tensor(evaluation, dtype=torch.float32) @ state
    # The first steps carry the error of discretizing the dynamics 1 / t where t is small.
    assert float((reconstructed - signal)[length // 4 :].abs().max()) < 0.05


def test_time_varying_memory_has_no_impulse_response() -> None:
    memory: HiPPOMemory = HiPPOMemory(HiPPOMeasure.LEGS, 4, 10, 1.0, Discretization.BILINEAR)
    with pytest.raises(ValueError, match="time-invariant"):
        memory.impulse_response(5)


@pytest.mark.parametrize("measure", list(HiPPOMeasure))
def test_measures_are_stable(measure: HiPPOMeasure) -> None:
    transition, input_matrix = measure.transition(8)
    assert transition.shape == (8, 8)
    assert input_matrix.shape == (8, 1)
    # The Fourier basis includes the constant sine of frequency 0, whose eigenvalue is 0.
    eigenvalues: Tensor = cast(Tensor, torch.linalg.eigvals(transition))
    assert bool((eigenvalues.real <= 1e-9).all())


def test_fourier_measure_needs_even_state() -> None:
    with pytest.raises(ValueError, match="even"):
        HiPPOMeasure.FOUT.transition(5)


def test_normal_plus_low_rank_rebuilds_legs() -> None:
    decomposition: NormalPlusLowRank = NormalPlusLowRank.from_legs(8)
    eigenvectors: Tensor = decomposition.eigenvectors
    normal: Tensor = eigenvectors @ torch.diag(decomposition.eigenvalues) @ eigenvectors.conj().T
    low_rank: Tensor = eigenvectors @ decomposition.low_rank
    transition, input_matrix = HiPPOMeasure.LEGS.transition(8)
    rebuilt: Tensor = normal - low_rank[:, None] * low_rank.conj()[None, :]
    assert torch.allclose(rebuilt, transition.to(torch.complex128), atol=1e-10)
    assert torch.allclose(eigenvectors @ decomposition.input_vector, input_matrix.squeeze(-1).to(torch.complex128))


@pytest.mark.parametrize("length", [10, 11])
def test_dplr_kernel_matches_recurrence(length: int) -> None:
    _ = torch.manual_seed(0)
    kernel_module: DPLRSSMKernel = DPLRSSMKernel(2, 8, 0.01, 0.1).double()
    with torch.no_grad():
        kernel: Tensor = kernel_module(length)
    evaluated_length: int = length if length % 2 == 1 else length + 1
    identity: Tensor = torch.eye(8, dtype=torch.complex128)
    for channel in range(2):
        eigenvalues: Tensor = torch.complex(
            -torch.exp(kernel_module.log_negative_real[channel]), kernel_module.imaginary[channel]
        )
        low_rank: Tensor = torch.view_as_complex(kernel_module.low_rank[channel].detach().contiguous())
        input_vector: Tensor = torch.view_as_complex(kernel_module.input_vector[channel].detach().contiguous())
        output_matrix: Tensor = torch.view_as_complex(kernel_module.output_matrix[channel].detach().contiguous())
        transition: Tensor = torch.diag(eigenvalues.detach()) - low_rank[:, None] * low_rank.conj()[None, :]
        step: float = float(torch.exp(kernel_module.log_step[channel].detach()))
        backward: Tensor = identity - step / 2.0 * transition
        discrete: Tensor = cast(Tensor, torch.linalg.solve(backward, identity + step / 2.0 * transition))
        discrete_input: Tensor = cast(Tensor, torch.linalg.solve(backward, step * input_vector))
        # The learned output matrix stands for conj(C) (I - A_bar^L) of the truncated kernel.
        truncation: Tensor = identity - cast(Tensor, torch.linalg.matrix_power(discrete, evaluated_length))
        output: Tensor = output_matrix.conj() @ cast(Tensor, torch.linalg.inv(truncation))
        expected: Tensor = torch.stack(
            [(output @ torch.linalg.matrix_power(discrete, lag) @ discrete_input).real for lag in range(length)]
        )
        assert torch.allclose(kernel[channel], expected, atol=1e-6)


@pytest.mark.parametrize("discretization", list(Discretization))
@pytest.mark.parametrize("initialization", list(DiagonalInitialization))
def test_diagonal_kernel_matches_recurrence(
    discretization: Discretization, initialization: DiagonalInitialization
) -> None:
    _ = torch.manual_seed(0)
    kernel_module: DiagonalSSMKernel = DiagonalSSMKernel(2, 8, initialization, discretization, 0.01, 0.1).double()
    with torch.no_grad():
        kernel: Tensor = kernel_module(12)
        eigenvalues: Tensor = torch.complex(-torch.exp(kernel_module.log_negative_real), kernel_module.imaginary)
        decay, input_scale = discretization.discretize_diagonal(eigenvalues, torch.exp(kernel_module.log_step)[:, None])
        output: Tensor = torch.view_as_complex(kernel_module.output_matrix.contiguous())
        expected: Tensor = torch.stack(
            [2.0 * (output * decay**lag * input_scale).sum(dim=-1).real for lag in range(12)], dim=-1
        )
    assert torch.allclose(kernel, expected, atol=1e-10)


@pytest.mark.parametrize("chunk_length", [1, 3, 4, 32])
def test_mamba2_chunks_match_sequential_recurrence(chunk_length: int) -> None:
    _ = torch.manual_seed(0)
    layer: Mamba2StateSpaceLayer = Mamba2StateSpaceLayer(8, 4, 2, 4, 2, chunk_length)
    batch_size, length, num_heads, head_size, state_size = 2, 10, 4, 4, 4
    inputs: Tensor = torch.randn(batch_size, length, num_heads, head_size)
    log_decays: Tensor = -torch.rand(batch_size, length, num_heads)
    input_matrix: Tensor = torch.randn(batch_size, length, state_size)
    output_matrix: Tensor = torch.randn(batch_size, length, state_size)
    with torch.no_grad():
        outputs: Tensor = layer._scan_chunks(inputs, log_decays, input_matrix, output_matrix)  # pyright: ignore[reportPrivateUsage]
    states: Tensor = sequential_recurrence(
        torch.exp(log_decays)[..., None, None].expand(-1, -1, -1, head_size, state_size).contiguous(),
        inputs[..., None] * input_matrix[:, :, None, None, :],
    )
    expected: Tensor = torch.einsum("blhpn,bln->blhp", states, output_matrix)
    assert torch.allclose(outputs, expected, atol=1e-4)
