import torch
from torch import Tensor

from time_series_dl.models import SequenceShape, TCNParameters, TSTParameters
from time_series_dl.models.architectures.pretrained import MaskedReconstructionParameters
from time_series_dl.models.components.masked_modeling import PatchMasking

ENCODER: TSTParameters = TSTParameters(model_size=8, num_heads=2, num_layers=1, feedforward_size=16)


def test_loss_is_the_error_of_the_hidden_values_only() -> None:
    _ = torch.manual_seed(0)
    network = MaskedReconstructionParameters(encoder=ENCODER).build_pretraining_network(
        SequenceShape(length=10, channels=2)
    )
    _ = network.eval()
    values: Tensor = torch.randn(3, 10, 2)
    is_hidden: Tensor = torch.rand(3, 10, 2) < 0.3
    with torch.no_grad():
        reconstructed: Tensor = network.decoder(network.encoder(values * ~is_hidden))
        expected: Tensor = (reconstructed - values)[is_hidden].square().mean()
        assert torch.allclose(network.compute_loss(values, is_hidden), expected, atol=1e-6)
        # Changing a hidden value changes only its own target, never the encoder's input.
        changed: Tensor = values + 10.0 * is_hidden
        changed_expected: Tensor = (reconstructed - changed)[is_hidden].square().mean()
        assert torch.allclose(network.compute_loss(changed, is_hidden), changed_expected, atol=1e-5)


def test_patch_masked_pretraining_trains_every_parameter() -> None:
    _ = torch.manual_seed(0)
    for encoder in (ENCODER, TCNParameters(channel_sizes=(4, 4))):
        model = MaskedReconstructionParameters(encoder=encoder, masking=PatchMasking(patch_length=3))
        network = model.build_pretraining_network(SequenceShape(length=12, channels=2))
        network(torch.randn(4, 12, 2)).backward()
        assert all(parameter.grad is not None for parameter in network.parameters())
