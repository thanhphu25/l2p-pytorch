import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from patch_transform import (
    ParametricFourQubitCircuit,
    PromptConditionedPatchTransform,
)
from prompt import Prompt
from result_summary import build_results_summary


class PhaseAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_original_l2p_is_the_default_path(self):
        prompt = Prompt(
            length=2, embed_dim=16, embedding_key='cls', prompt_pool=True,
            prompt_key=True, pool_size=5, top_k=2, batchwise_prompt=False)
        patches = torch.randn(3, 16, 16)
        queries = torch.randn(3, 16)
        output = prompt(patches, cls_features=queries)

        expected = output['similarity'].topk(2, dim=1).indices
        self.assertTrue(torch.equal(output['prompt_idx'], expected))
        self.assertFalse(hasattr(prompt, 'patch_transformer'))
        self.assertNotIn('patch_entropy', output)

    def test_circuit_probabilities_are_normalized_and_differentiable(self):
        circuit = ParametricFourQubitCircuit(depth=2, init_std=0.1)
        phases = torch.randn(3, 16, requires_grad=True)
        probabilities = circuit(phases)

        self.assertTrue(torch.all(probabilities >= 0.0))
        self.assertTrue(torch.allclose(
            probabilities.sum(1), torch.ones(3), atol=1e-6))
        probabilities[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(phases.grad).all())
        self.assertTrue(torch.isfinite(circuit.angles.grad).all())

    def test_no_encoding_removes_image_and_prompt_conditioning(self):
        transform = PromptConditionedPatchTransform(
            embed_dim=16, mode='phase_no_encoding', latent_dim=8)
        patches_a = torch.randn(2, 196, 16)
        patches_b = torch.randn(2, 196, 16)
        prompts_a = torch.randn(2, 5, 5, 16)
        prompts_b = torch.randn(2, 5, 5, 16)
        modulated, diagnostics_a = transform(patches_a, prompts_a)
        _, diagnostics_b = transform(patches_b, prompts_b)

        self.assertTrue(torch.allclose(
            diagnostics_a['patch_probabilities'],
            diagnostics_b['patch_probabilities'], atol=1e-6))
        self.assertEqual(diagnostics_a['phase_std'].item(), 0.0)
        modulated.square().mean().backward()
        projection_grad = transform.patch_phase_projection.weight.grad
        self.assertIsNotNone(projection_grad)
        self.assertEqual(projection_grad.abs().max().item(), 0.0)

    def test_mlp_and_phase_parameters_are_matched(self):
        phase = PromptConditionedPatchTransform(
            embed_dim=768, mode='phase', latent_dim=32, circuit_depth=2)
        mlp = PromptConditionedPatchTransform(
            embed_dim=768, mode='mlp', latent_dim=32, circuit_depth=2)
        phase_count = sum(parameter.numel() for parameter in phase.parameters())
        mlp_count = sum(parameter.numel() for parameter in mlp.parameters())

        self.assertEqual(phase_count, 49169)
        self.assertEqual(mlp_count, 49218)
        self.assertLess(abs(phase_count - mlp_count) / phase_count, 0.002)

    def test_phase_integrates_after_original_prompt_retrieval(self):
        prompt = Prompt(
            length=2, embed_dim=16, embedding_key='cls', prompt_pool=True,
            prompt_key=True, pool_size=5, top_k=2, batchwise_prompt=False,
            patch_transform='phase', phase_latent_dim=8)
        patches = torch.randn(3, 16, 16)
        queries = torch.randn(3, 16)
        output = prompt(patches, cls_features=queries)

        self.assertEqual(output['prompted_embedding'].shape, (3, 20, 16))
        self.assertGreater(output['phase_std'].item(), 0.0)
        output['prompted_embedding'].square().mean().backward()
        gradient = prompt.patch_transformer.patch_phase_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_ablation_summary_is_json_serializable(self):
        args = SimpleNamespace(
            patch_transform='phase', dataset='Split-CIFAR100', seed=10961,
            batchwise_prompt=True, num_tasks=2, phase_latent_dim=32,
            phase_circuit_depth=2, phase_alpha_init=0.05,
            phase_alpha_max=0.2, phase_circuit_init_std=0.1)
        tasks = [
            {
                'task': 1, 'avg_acc1': 90.0, 'avg_acc5': 99.0,
                'avg_loss': 0.3, 'forgetting': 0.0,
                'backward_transfer': 0.0, 'old_task_acc1': None,
                'current_task_acc1': 90.0, 'old_new_gap': None,
                'train_patch_metrics': {},
            },
            {
                'task': 2, 'avg_acc1': 85.0, 'avg_acc5': 98.0,
                'avg_loss': 0.5, 'forgetting': 6.0,
                'backward_transfer': -6.0, 'old_task_acc1': 84.0,
                'current_task_acc1': 86.0, 'old_new_gap': 2.0,
                'train_patch_metrics': {'PatchDev': 0.01},
            },
        ]
        summary = build_results_summary(
            args, tasks, np.array([[90.0, 84.0], [0.0, 86.0]]),
            status='completed')

        self.assertEqual(summary['method'], 'L2P + Phase Interference')
        self.assertEqual(summary['avg_incremental_acc'], 87.5)
        self.assertEqual(summary['final_old_new_gap'], 2.0)
        json.dumps(summary)


if __name__ == '__main__':
    unittest.main()
