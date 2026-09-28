import pytest

torch = pytest.importorskip('torch')

from input_prompt import InputSpatialPrompt


@pytest.mark.parametrize('router', ['cosine', 'linear', 'quantum', 'quantum_no_phase'])
def test_forward_backward_and_normalized_weights(router):
    module = InputSpatialPrompt(
        query_dim=32,
        pool_size=10,
        top_k=5,
        hidden_dim=4,
        router=router,
        global_prompt=True,
    )
    image = torch.randn(3, 3, 32, 32, requires_grad=True)
    query = torch.randn(3, 32)

    prompted, diagnostics = module(image, query)

    assert prompted.shape == image.shape
    assert diagnostics['input_prompt_idx'].shape == (3, 5)
    assert torch.allclose(
        diagnostics['input_prompt_weights'].sum(dim=-1),
        torch.ones(3),
        atol=1e-5,
    )
    prompted.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_uniform_frequency_filter_starts_as_identity_residual():
    module = InputSpatialPrompt(
        query_dim=16,
        pool_size=4,
        top_k=2,
        router='cosine',
        global_prompt=True,
        frequency_rings=4,
    )
    image = torch.randn(2, 3, 32, 32)
    residual = module._frequency_prompt(image)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)
