import pytest
import torch
from anomaly_network_checks import assert_gradients_close, check_anomaly_scoring_network, collect_gradients
from torch import Tensor

from time_series_dl.models import SequenceShape
from time_series_dl.models.architectures.transformer.dcdetector import (
    DCdetectorAnomalyScoringNetwork,
    DCdetectorParameters,
)

SHAPE: SequenceShape = SequenceShape(length=12, channels=2)
MODEL: DCdetectorParameters = DCdetectorParameters(patch_sizes=(2, 3), model_size=8, num_heads=2, num_layers=2)


def build() -> DCdetectorAnomalyScoringNetwork:
    _ = torch.manual_seed(0)
    network = MODEL.build_anomaly_scoring_network(SHAPE)
    assert isinstance(network, DCdetectorAnomalyScoringNetwork)
    return network


@pytest.mark.parametrize(
    ("model", "shape"),
    [
        (MODEL, SHAPE),
        (
            DCdetectorParameters(patch_sizes=(1, 5), model_size=4, num_heads=1, num_layers=1),
            SequenceShape(length=5, channels=1),
        ),
    ],
)
def test_anomaly_scoring_network(model: DCdetectorParameters, shape: SequenceShape) -> None:
    _ = check_anomaly_scoring_network(model, shape)


def test_associations_follow_the_patch_structure() -> None:
    network = build().eval()
    with torch.no_grad():
        patch_wise, in_patch = network.associate(torch.rand(2, SHAPE.length, SHAPE.channels))
    assert len(patch_wise) == len(in_patch) == len(MODEL.patch_sizes) * MODEL.num_layers
    for index, (n, p) in enumerate(zip(patch_wise, in_patch, strict=True)):
        size: int = MODEL.patch_sizes[index // MODEL.num_layers]
        assert n.shape == p.shape == (2, MODEL.num_heads, SHAPE.length, SHAPE.length)
        assert torch.allclose(n.sum(dim=-1), torch.ones(2, MODEL.num_heads, SHAPE.length))
        assert torch.allclose(p.sum(dim=-1), torch.ones(2, MODEL.num_heads, SHAPE.length))
        # Patch-wise: constant over the steps of each pair of patches. In-patch: periodic in the patch length.
        assert torch.allclose(n, n[..., ::size, ::size].repeat_interleave(size, -2).repeat_interleave(size, -1))
        assert torch.allclose(p, p[..., :size, :size].repeat(1, 1, SHAPE.length // size, SHAPE.length // size))


def test_contrastive_loss_pulls_the_in_patch_view_and_pushes_the_patch_wise_view() -> None:
    network = build()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    _ = network.eval()
    patch_wise, in_patch = network.associate(values)
    pairs = list(zip(patch_wise, in_patch, strict=True))
    discrepancy: Tensor = torch.stack([network.compute_discrepancy(p, n).mean() for n, p in pairs]).mean()
    in_patch_loss: Tensor = torch.stack([network.compute_discrepancy(p, n.detach()).mean() for n, p in pairs]).mean()
    patch_wise_loss: Tensor = torch.stack([network.compute_discrepancy(p.detach(), n).mean() for n, p in pairs]).mean()
    loss: Tensor = network.compute_loss(values, epoch=0)
    assert loss.item() == pytest.approx(-discrepancy.item(), rel=1e-5)
    assert_gradients_close(
        collect_gradients(loss, network, "in_patch_embeddings"),
        collect_gradients(in_patch_loss, network, "in_patch_embeddings"),
    )
    assert_gradients_close(
        collect_gradients(loss, network, "patch_embeddings"),
        collect_gradients(-patch_wise_loss, network, "patch_embeddings"),
    )


def test_score_is_the_softmax_of_the_negative_discrepancy() -> None:
    network = build().eval()
    values: Tensor = torch.rand(2, SHAPE.length, SHAPE.channels)
    with torch.no_grad():
        patch_wise, in_patch = network.associate(values)
        discrepancy: Tensor = torch.stack(
            [network.compute_discrepancy(p, n) for n, p in zip(patch_wise, in_patch, strict=True)]
        ).sum(dim=0)
        expected: Tensor = torch.softmax(-MODEL.score_temperature * discrepancy, dim=-1)
        assert torch.allclose(network(values), expected)


def test_parameters_validate() -> None:
    with pytest.raises(ValueError, match="empty"):
        _ = DCdetectorParameters(patch_sizes=())
    with pytest.raises(ValueError, match="repeat"):
        _ = DCdetectorParameters(patch_sizes=(3, 3))
    with pytest.raises(ValueError, match="num_heads"):
        _ = DCdetectorParameters(model_size=6, num_heads=4)
    with pytest.raises(ValueError, match="do not divide the window length 10"):
        _ = MODEL.build_anomaly_scoring_network(SequenceShape(length=10, channels=1))
