import unittest

import torch

from prompt import Prompt, QuantumStateRouter


class QuantumStateRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_density_states_and_measurement_are_valid(self):
        router = QuantumStateRouter(
            pool_size=5, embed_dim=12, state_dim=4, rank=3, memory_size=3)
        keys = torch.randn(5, 12)
        queries = torch.randn(6, 12)
        patches = torch.randn(6, 8, 12)

        prompt_states = router.prompt_states(keys)
        image_states = router.image_states(queries, patches)
        measurement = router.pretty_good_measurement(prompt_states)
        probabilities = router.measure(measurement, image_states)

        self.assertTrue(torch.allclose(
            prompt_states.diagonal(dim1=-2, dim2=-1).sum(-1),
            torch.ones(5), atol=1e-5))
        self.assertTrue(torch.allclose(
            image_states.diagonal(dim1=-2, dim2=-1).sum(-1),
            torch.ones(6), atol=1e-5))
        self.assertGreaterEqual(torch.linalg.eigvalsh(prompt_states).min().item(), -1e-5)
        self.assertGreaterEqual(torch.linalg.eigvalsh(image_states).min().item(), -1e-5)
        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.allclose(
            probabilities.sum(1), torch.ones(6), atol=1e-5))

    def test_qsd_starts_from_cosine_topk(self):
        prompt = Prompt(
            length=2, embed_dim=12, embedding_key='cls', prompt_pool=True,
            prompt_key=True, pool_size=5, top_k=2, batchwise_prompt=False,
            prompt_router='qsd', qsd_state_dim=4, qsd_rank=2)
        patches = torch.randn(4, 8, 12)
        queries = torch.randn(4, 12)

        output = prompt(
            patches, cls_features=queries, prompt_query_tokens=patches)
        cosine_topk = output['similarity'].topk(2, dim=1).indices

        self.assertEqual(prompt.quantum_router.residual_strength.item(), 0.0)
        self.assertTrue(torch.equal(output['prompt_idx'], cosine_topk))
        self.assertTrue(torch.isfinite(output['prompted_embedding']).all())

    def test_default_router_remains_plain_l2p_cosine(self):
        prompt = Prompt(
            length=2, embed_dim=12, embedding_key='cls', prompt_pool=True,
            prompt_key=True, pool_size=5, top_k=2, batchwise_prompt=False)
        patches = torch.randn(4, 8, 12)
        queries = torch.randn(4, 12)
        output = prompt(patches, cls_features=queries)

        expected = output['similarity'].topk(2, dim=1).indices
        self.assertTrue(torch.equal(output['prompt_idx'], expected))
        self.assertNotIn('qsd_probabilities', output)
        self.assertFalse(hasattr(prompt, 'quantum_router'))

    def test_router_receives_gradient_and_retains_task_state(self):
        prompt = Prompt(
            length=2, embed_dim=12, embedding_key='cls', prompt_pool=True,
            prompt_key=True, pool_size=5, top_k=2, batchwise_prompt=False,
            prompt_router='qsd', qsd_state_dim=4, qsd_rank=2,
            qsd_memory_size=3)
        patches = torch.randn(4, 8, 12)
        queries = torch.randn(4, 12)

        output = prompt(
            patches, cls_features=queries, prompt_query_tokens=patches,
            update_router_memory=True)
        weights = torch.randn_like(output['prompted_embedding'])
        (output['prompted_embedding'] * weights).sum().backward()
        self.assertIsNotNone(prompt.quantum_router.residual_strength.grad)
        self.assertTrue(torch.isfinite(prompt.quantum_router.residual_strength.grad))
        self.assertNotEqual(prompt.quantum_router.residual_strength.grad.item(), 0.0)

        prompt.consolidate_router()
        self.assertEqual(prompt.quantum_router.anchor_count.item(), 1)
        self.assertEqual(prompt.quantum_router.pending_count.item(), 0)

        with torch.no_grad():
            prompt.quantum_router.residual_strength.fill_(0.1)
        prompt.zero_grad(set_to_none=True)
        output = prompt(
            patches, cls_features=queries, prompt_query_tokens=patches)
        output['prompted_embedding'].square().mean().backward()
        self.assertIsNotNone(prompt.quantum_router.projection.weight.grad)
        self.assertTrue(torch.isfinite(prompt.quantum_router.projection.weight.grad).all())
        self.assertTrue(torch.isfinite(output['qsd_retention_loss']))


if __name__ == '__main__':
    unittest.main()
