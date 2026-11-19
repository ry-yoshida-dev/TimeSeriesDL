import pytest
import torch
from torch import Tensor

from time_series_dl.models.components.graph import (
    AdaptiveAdjacency,
    ChebyshevPolynomials,
    DiffusionConvolution,
    DirectedGraphLearner,
    MixHopPropagation,
    NodeAdaptiveGraphConvolution,
)


@pytest.mark.parametrize("is_symmetric", [False, True])
def test_adaptive_adjacency_is_row_softmax_of_embedding_scores(is_symmetric: bool) -> None:
    _ = torch.manual_seed(0)
    adjacency: AdaptiveAdjacency = AdaptiveAdjacency(5, 3, is_symmetric=is_symmetric)
    source: Tensor = adjacency.node_embeddings
    target: Tensor = source if is_symmetric else adjacency.embeddings[1]
    expected: Tensor = torch.softmax(torch.relu(source @ target.T), dim=1)
    with torch.no_grad():
        assert torch.allclose(adjacency(), expected)
        assert torch.allclose(adjacency().sum(dim=1), torch.ones(5))


@pytest.mark.parametrize(("num_nodes", "num_neighbors"), [(1, 3), (6, 2), (6, 10)])
def test_directed_graph_learner_is_sparse_and_uni_directed(num_nodes: int, num_neighbors: int) -> None:
    _ = torch.manual_seed(0)
    learner: DirectedGraphLearner = DirectedGraphLearner(num_nodes, 4, num_neighbors, saturation=3.0).eval()
    with torch.no_grad():
        adjacency: Tensor = learner()
    assert adjacency.shape == (num_nodes, num_nodes)
    assert bool((adjacency >= 0).all())
    assert bool(((adjacency > 0).sum(dim=1) <= num_neighbors).all())
    assert bool((adjacency * adjacency.T == 0).all())
    with torch.no_grad():
        assert torch.equal(learner(), adjacency)


def test_mix_hop_propagation_matches_explicit_hops() -> None:
    _ = torch.manual_seed(0)
    propagation: MixHopPropagation = MixHopPropagation(2, 3, depth=2, retain_ratio=0.1)
    features: Tensor = torch.randn(2, 2, 4, 5)
    adjacency: Tensor = torch.rand(4, 4)
    looped: Tensor = adjacency + torch.eye(4)
    transition: Tensor = looped / looped.sum(dim=1, keepdim=True)
    first: Tensor = 0.1 * features + 0.9 * torch.einsum("vw,bcwt->bcvt", transition, features)
    second: Tensor = 0.1 * features + 0.9 * torch.einsum("vw,bcwt->bcvt", transition, first)
    with torch.no_grad():
        expected: Tensor = propagation.selection(torch.cat([features, first, second], dim=1))
        assert torch.allclose(propagation(features, adjacency), expected, atol=1e-6)


def test_diffusion_convolution_matches_matrix_powers() -> None:
    _ = torch.manual_seed(0)
    convolution: DiffusionConvolution = DiffusionConvolution(2, 3, num_supports=2, order=2, dropout=0.0)
    features: Tensor = torch.randn(2, 2, 4, 5)
    supports: tuple[Tensor, Tensor] = (torch.rand(4, 4), torch.rand(4, 4))
    diffused: list[Tensor] = [features]
    for support in supports:
        for power in (1, 2):
            diffused.append(torch.einsum("vw,bcwt->bcvt", torch.linalg.matrix_power(support, power), features))
    with torch.no_grad():
        expected: Tensor = convolution.mixture(torch.cat(diffused, dim=1))
        assert torch.allclose(convolution(features, supports), expected, atol=1e-5)
    with pytest.raises(ValueError, match="supports"):
        convolution(features, supports[:1])


def test_chebyshev_polynomials_match_cosine_form() -> None:
    values: Tensor = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64)
    matrices: Tensor = torch.stack([torch.diag(values), torch.diag(values.flip(0))])
    polynomials: Tensor = ChebyshevPolynomials(num_terms=5).evaluate(matrices)
    assert polynomials.shape == (2, 5, 7, 7)
    for degree in range(5):
        expected: Tensor = torch.cos(degree * torch.arccos(values))
        assert torch.allclose(torch.diagonal(polynomials[0, degree]), expected, atol=1e-9)
    assert ChebyshevPolynomials(num_terms=1).evaluate(torch.rand(3, 3)).shape == (1, 3, 3)
    with pytest.raises(ValueError, match="square"):
        ChebyshevPolynomials(num_terms=2).evaluate(torch.rand(3, 4))
    with pytest.raises(ValueError, match="num_terms"):
        ChebyshevPolynomials(num_terms=0)


def test_node_adaptive_graph_convolution_matches_per_node_weights() -> None:
    _ = torch.manual_seed(0)
    convolution: NodeAdaptiveGraphConvolution = NodeAdaptiveGraphConvolution(2, 3, num_supports=2, embedding_size=4)
    features: Tensor = torch.randn(2, 5, 2)
    supports: Tensor = torch.rand(2, 5, 5)
    embeddings: Tensor = torch.randn(5, 4)
    with torch.no_grad():
        outputs: Tensor = convolution(features, supports, embeddings)
        for node in range(5):
            expected: Tensor = embeddings[node] @ convolution.bias_pool
            for term in range(2):
                weight: Tensor = torch.einsum("d,dio->io", embeddings[node], convolution.weight_pool[:, term])
                expected = expected + (supports[term, node] @ features) @ weight
            assert torch.allclose(outputs[:, node], expected, atol=1e-5)


def test_components_reject_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        AdaptiveAdjacency(0, 3, is_symmetric=True)
    with pytest.raises(ValueError, match="saturation"):
        DirectedGraphLearner(3, 2, 2, saturation=0.0)
    with pytest.raises(ValueError, match="retain_ratio"):
        MixHopPropagation(2, 2, depth=1, retain_ratio=-0.1)
    with pytest.raises(ValueError, match="dropout"):
        DiffusionConvolution(2, 2, 1, 1, dropout=1.0)
    with pytest.raises(ValueError, match="embedding_size"):
        NodeAdaptiveGraphConvolution(2, 2, 1, embedding_size=0)
