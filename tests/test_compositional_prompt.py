import inspect
import io
import unittest

import torch
from torch.nn import functional as F

from compositional_prompt import CompositionalPrompt, _InverseSqrt, complete_pgm

# torch<1.13 (the Kaggle pin is 1.12.1) has no weights_only argument.
WEIGHTS_ONLY = ({'weights_only': True}
                if 'weights_only' in inspect.signature(torch.load).parameters else {})


class CompositionalPromptTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)
        torch.set_num_threads(1)

    def make_prompt(self, **kwargs):
        options = dict(embed_dim=12, length=2, heads=2, num_tasks=3,
                       components_per_task=3, num_classes=6, state_dim=4,
                       rank=2, prototypes_per_class=2)
        options.update(kwargs)
        return CompositionalPrompt(**options)

    def inputs(self, batch=8):
        return torch.randn(batch, 6, 12), torch.randn(batch, 12)

    def run_prompt(self, prompt, patches, queries):
        return prompt(patches, cls_features=queries, prompt_query_tokens=patches)

    def memorize(self, prompt, label=0):
        patches, queries = self.inputs()
        query, density = prompt.encode_states(queries, patches)
        prompt.consolidate_class(label, query, density)

    def test_complete_povm_in_rank_deficient_space(self):
        factors = torch.randn(2, 3, 8, 1, dtype=torch.double, requires_grad=True)
        states = factors @ factors.mT
        states = states / states.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        operators = complete_pgm(states, 1e-3)
        torch.testing.assert_close(operators.sum(1), torch.eye(8, dtype=torch.double).expand(2, 8, 8))
        self.assertGreater(torch.linalg.eigvalsh(operators).min().item(), -1e-10)
        weights = torch.randn_like(operators)
        (weights * operators).sum().backward()
        self.assertTrue(torch.isfinite(factors.grad).all())
        self.assertGreater(factors.grad.abs().sum().item(), 0)

    def test_inverse_sqrt_derivative_at_repeated_eigenvalues(self):
        matrix = (torch.eye(4, dtype=torch.double) * 0.2).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(_InverseSqrt.apply, (matrix,), atol=1e-5))
        random = torch.randn(4, 4, dtype=torch.double)
        matrix = (random @ random.T + 0.1 * torch.eye(4)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(_InverseSqrt.apply, (matrix,), atol=1e-5))

    def test_density_valid_including_zero_features(self):
        prompt = self.make_prompt()
        patches, queries = self.inputs()
        patches[0], queries[0] = 0, 0
        query, density = prompt.encode_states(queries, patches)
        torch.testing.assert_close(query.norm(dim=-1), torch.ones(8))
        torch.testing.assert_close(density.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(8))
        self.assertGreater(torch.linalg.eigvalsh(density).min().item(), -1e-6)
        self.assertNotIn('projection', dict(prompt.named_parameters()))

    def test_composition_keeps_patch_tokens_and_ce_trains_router(self):
        prompt = self.make_prompt(quantum_mix=1.0)
        patches, queries = self.inputs()
        output = self.run_prompt(prompt, patches, queries)
        tokens = output['prompted_embedding']
        self.assertEqual(tokens.shape, (8, 10, 12))
        self.assertTrue(torch.equal(tokens[:, 4:], patches))
        torch.testing.assert_close(output['route_probabilities'].sum(-1), torch.ones(8, 2))
        classifier = torch.nn.Linear(48, 6)
        loss = F.cross_entropy(classifier(tokens[:, :4].flatten(1)), torch.arange(8) % 6)
        loss.backward()
        for parameter in (prompt.prompts[0], prompt.factors[0]):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 1e-8)

    def test_routing_is_per_image_not_batch_majority(self):
        prompt = self.make_prompt().eval()
        patches, queries = self.inputs()
        alone = self.run_prompt(prompt, patches[:1], queries[:1])['route_probabilities']
        together = self.run_prompt(prompt, patches, queries)['route_probabilities'][:1]
        torch.testing.assert_close(alone, together)

    def test_frozen_old_bank_survives_optimizer_momentum_and_decay(self):
        prompt = self.make_prompt()
        optimizer = torch.optim.AdamW(prompt.parameters(), lr=0.01, weight_decay=0.1)
        patches, queries = self.inputs()
        self.run_prompt(prompt, patches, queries)['prompted_embedding'].square().mean().backward()
        optimizer.step()
        old = [p.detach().clone() for p in (prompt.prompts[0], prompt.factors[0])]
        prompt.begin_task(1)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            self.run_prompt(prompt, patches, queries)['prompted_embedding'].square().mean().backward()
            optimizer.step()
        for expected, actual in zip(old, (prompt.prompts[0], prompt.factors[0])):
            self.assertTrue(torch.equal(expected, actual))
            self.assertIsNone(actual.grad)
            self.assertFalse(actual.requires_grad)

    def test_targets_are_final_router_snapshots_and_remain_immutable(self):
        prompt = self.make_prompt()
        self.memorize(prompt)
        valid = prompt.memory_valid
        target = prompt.memory_probs.clone()
        probs = prompt.route(prompt.memory_queries[valid], prompt.memory_states[valid])[0]
        torch.testing.assert_close(probs, target[valid, :, :3])
        prompt.begin_task(1)
        self.memorize(prompt, label=1)
        self.assertTrue(torch.equal(prompt.memory_probs[:2], target[:2]))
        self.assertTrue(torch.equal(prompt.memory_probs[:2, :, 3:], torch.zeros_like(target[:2, :, 3:])))
        with self.assertRaises(ValueError):
            self.memorize(prompt, label=0)

    def test_retention_penalizes_new_mass_and_improves_when_optimized(self):
        prompt = self.make_prompt()
        self.memorize(prompt)
        prompt.begin_task(1)
        prompt.eval()
        optimizer = torch.optim.Adam([prompt.factors[1]], lr=0.02)
        def loss_values():
            return prompt.retention_loss(prompt.routing_operators(), prompt.prompt_components())
        first = loss_values()
        self.assertGreater(first[0].item(), 0)
        self.assertGreater(first[2].item(), 0)
        for _ in range(25):
            optimizer.zero_grad()
            losses = loss_values()
            (losses[0] + losses[1]).backward()
            optimizer.step()
        last = loss_values()
        self.assertLess((last[0] + last[1]).item(), (first[0] + first[1]).item())
        self.assertLess(last[2].item(), first[2].item())

    def test_checkpoint_restores_seen_capacity_memory_and_freezing(self):
        prompt = self.make_prompt()
        self.memorize(prompt)
        prompt.begin_task(1)
        self.memorize(prompt, label=1)
        stream = io.BytesIO()
        torch.save(prompt.state_dict(), stream)
        stream.seek(0)
        restored = self.make_prompt()
        restored.load_state_dict(torch.load(stream, **WEIGHTS_ONLY))
        patches, queries = self.inputs()
        prompt.eval()
        restored.eval()
        expected = self.run_prompt(prompt, patches, queries)
        actual = self.run_prompt(restored, patches, queries)
        for name in expected:
            if torch.is_tensor(expected[name]):
                torch.testing.assert_close(actual[name], expected[name])
        self.assertEqual([p.requires_grad for p in restored.prompts], [False, True, False])
        self.assertEqual(restored.active_tasks.item(), 2)

    def test_classical_control_matches_zero_quantum_mix_and_capacity(self):
        quantum = self.make_prompt(quantum_mix=0)
        classical = self.make_prompt(router='cosine_comp')
        classical.load_state_dict(quantum.state_dict())
        patches, queries = self.inputs()
        expected = self.run_prompt(quantum, patches, queries)
        actual = self.run_prompt(classical, patches, queries)
        torch.testing.assert_close(actual['prompted_embedding'], expected['prompted_embedding'])
        self.assertEqual(sum(p.numel() for p in quantum.parameters()),
                         sum(p.numel() for p in classical.parameters()))
        actual['prompted_embedding'].square().mean().backward()
        self.assertTrue((classical.factors[0].grad.abs().sum((0, 1, 2)) > 0).all())

    def test_invalid_mask_and_out_of_order_task_are_rejected(self):
        prompt = self.make_prompt()
        patches, queries = self.inputs()
        with self.assertRaises(ValueError):
            prompt.begin_task(2)
        with self.assertRaises(ValueError):
            prompt(patches, prompt_mask=torch.tensor([[0]]), cls_features=queries,
                   prompt_query_tokens=patches)


if __name__ == '__main__':
    unittest.main()
