"""Batches and checks shared by the tests of every irregular network."""

from pathlib import Path
from typing import cast

import torch
import yaml
from torch import Tensor

import time_series_dl
from time_series_dl.models.interfaces.irregular_interpolation_network import IrregularInterpolationNetwork
from time_series_dl.models.interfaces.irregular_summary_network import IrregularSummaryNetwork
from time_series_dl.models.interfaces.irregular_tensors import IrregularTensors
from time_series_dl.models.interfaces.protocols.irregular_interpolation import IrregularInterpolationModel
from time_series_dl.models.interfaces.protocols.irregular_summary import IrregularSummaryModel
from time_series_dl.models.interfaces.shapes.irregular_sequence import IrregularSequenceShape
from time_series_dl.models.interfaces.shapes.irregular_summary import IrregularSummaryShape

CHANNELS: int = 3
LENGTHS: tuple[int, ...] = (7, 4, 1)


def make_sequences(lengths: tuple[int, ...] = LENGTHS, channels: int = CHANNELS, seed: int = 0) -> IrregularTensors:
    """
    Build a padded batch of random irregular sequences in the convention of ``IrregularTensors``.

    Parameters
    ----------
    lengths : tuple[int, ...]
        Number of real steps of each sample.
    channels : int
        Number of channels.
    seed : int
        Seed of the random values.

    Returns
    -------
    IrregularTensors
        Sequences whose steps observe at least one channel, padded to the longest.
    """
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    max_length: int = max(lengths)
    times: Tensor = torch.rand(len(lengths), max_length, generator=generator).sort(dim=1).values
    mask: Tensor = (torch.rand(len(lengths), max_length, channels, generator=generator) < 0.6).float()
    mask[:, :, 0] = torch.maximum(mask[:, :, 0], (mask.sum(dim=-1) == 0).float())
    values: Tensor = torch.randn(len(lengths), max_length, channels, generator=generator)
    for index, length in enumerate(lengths):
        times[index, length:] = times[index, length - 1]
        mask[index, length:] = 0.0
    return IrregularTensors(
        times=times, values=values * mask, observed_mask=mask, lengths=torch.tensor(lengths, dtype=torch.int64)
    )


def pad_more(sequences: IrregularTensors, extra: int) -> IrregularTensors:
    """
    Append padding steps that repeat each sample's last time and observe nothing.

    Parameters
    ----------
    sequences : IrregularTensors
        The sequences.
    extra : int
        Number of padding steps to append.

    Returns
    -------
    IrregularTensors
        The longer sequences.
    """
    last_times: Tensor = sequences.times.gather(1, (sequences.lengths - 1).unsqueeze(1)).expand(-1, extra)
    zeros: Tensor = sequences.values.new_zeros(sequences.values.shape[0], extra, sequences.channels)
    return IrregularTensors(
        times=torch.cat([sequences.times, last_times], dim=1),
        values=torch.cat([sequences.values, zeros], dim=1),
        observed_mask=torch.cat([sequences.observed_mask, zeros], dim=1),
        lengths=sequences.lengths,
    )


def take(sequences: IrregularTensors, index: int) -> IrregularTensors:
    """
    Take one sample, trimmed to its own length.

    Parameters
    ----------
    sequences : IrregularTensors
        The sequences.
    index : int
        The sample.

    Returns
    -------
    IrregularTensors
        A batch of that sample alone.
    """
    return IrregularTensors(
        times=sequences.times[index : index + 1],
        values=sequences.values[index : index + 1],
        observed_mask=sequences.observed_mask[index : index + 1],
        lengths=sequences.lengths[index : index + 1],
    ).trimmed()


def check_irregular_summary_network(model: IrregularSummaryModel, output_size: int = 2) -> IrregularSummaryNetwork:
    """
    Check output shapes, gradients of every parameter, and invariance to padding.

    Parameters
    ----------
    model : IrregularSummaryModel
        The model under test.
    output_size : int
        Size of the outputs.

    Returns
    -------
    IrregularSummaryNetwork
        The network, in evaluation mode.
    """
    _ = torch.manual_seed(0)
    network: IrregularSummaryNetwork = model.build_irregular_summary_network(
        IrregularSummaryShape(input_channels=CHANNELS, output_size=output_size)
    )
    sequences: IrregularTensors = make_sequences()
    _ = network.train()
    loss: Tensor = network.compute_training_loss(sequences, lambda outputs: outputs.square().mean())
    loss.backward()
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    with torch.no_grad():
        outputs: Tensor = network(sequences)
        padded: Tensor = network(pad_more(sequences, 5))
        alone: list[Tensor] = [network(take(sequences, index)) for index in range(len(LENGTHS))]
    assert outputs.shape == (len(LENGTHS), output_size)
    assert bool(torch.isfinite(outputs).all())
    assert torch.allclose(outputs, padded, atol=1e-5)
    assert torch.allclose(outputs, torch.cat(alone), atol=1e-5)
    return network


def check_irregular_interpolation_network(model: IrregularInterpolationModel) -> IrregularInterpolationNetwork:
    """
    Check output shapes, gradients of every parameter, and invariance to padding of the context and the queries.

    Parameters
    ----------
    model : IrregularInterpolationModel
        The model under test.

    Returns
    -------
    IrregularInterpolationNetwork
        The network, in evaluation mode.
    """
    _ = torch.manual_seed(0)
    network: IrregularInterpolationNetwork = model.build_irregular_interpolation_network(
        IrregularSequenceShape(channels=CHANNELS)
    )
    context: IrregularTensors = make_sequences(seed=1)
    targets: IrregularTensors = make_sequences(lengths=(3, 5, 2), seed=2)
    _ = network.train()
    loss: Tensor = network.compute_training_loss(context, targets)
    loss.backward()
    assert bool(torch.isfinite(loss))
    missing: list[str] = [name for name, parameter in network.named_parameters() if parameter.grad is None]
    assert not missing, f"parameters without gradient: {missing}"
    _ = network.eval()
    query_times: Tensor = torch.linspace(0.0, 1.5, 6).repeat(len(LENGTHS), 1)
    with torch.no_grad():
        outputs: Tensor = network(context, query_times)
        padded: Tensor = network(pad_more(context, 4), torch.cat([query_times, query_times[:, -1:].repeat(1, 3)], 1))
        alone: list[Tensor] = [
            network(take(context, index), query_times[index : index + 1]) for index in range(len(LENGTHS))
        ]
    assert outputs.shape == (len(LENGTHS), 6, CHANNELS)
    assert bool(torch.isfinite(outputs).all())
    assert torch.allclose(outputs, padded[:, :6], atol=1e-5)
    assert torch.allclose(padded[:, 6:], padded[:, 5:6].expand(-1, 3, -1), atol=1e-5)
    assert torch.allclose(outputs, torch.cat(alone), atol=1e-5)
    return network


def read_preset_parameters(architecture: str) -> dict[str, object]:
    """
    Read the parameters of a YAML preset of ``config/``.

    Parameters
    ----------
    architecture : str
        The architecture value, which names the preset file.

    Returns
    -------
    dict[str, object]
        The ``model.parameters`` section.
    """
    document: object = yaml.safe_load(
        (Path(time_series_dl.__file__).parent / "config" / f"{architecture}.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(document, dict)
    # YAML mappings are untyped, so give the keys and values the widest type.
    section: object = cast(dict[object, object], document)["model"]
    assert isinstance(section, dict)
    model: dict[object, object] = cast(dict[object, object], section)
    assert model["architecture"] == architecture
    parameters: object = model["parameters"]
    assert isinstance(parameters, dict)
    return {str(key): value for key, value in cast(dict[object, object], parameters).items()}
