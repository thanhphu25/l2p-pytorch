import argparse
import inspect
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from timm.optim import create_optimizer

from configs.cifar100_l2p import get_args_parser
from drift_compensation import HeisenbergHeads, fit_drift
from engine import train_and_evaluate
from result_summary import matrix_metrics
from vision_transformer import VisionTransformer

# Checkpoints pickle argparse args; torch<1.13 (Kaggle pin 1.12.1) has no weights_only.
FULL_PICKLE = ({'weights_only': False}
               if 'weights_only' in inspect.signature(torch.load).parameters else {})


def orthogonal(dim):
    return torch.linalg.qr(torch.randn(dim, dim, dtype=torch.double))[0]


class FitDriftTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)

    def test_unitary_recovers_a_rigid_drift_exactly(self):
        before = torch.randn(200, 6, dtype=torch.double)
        after = before @ orthogonal(6) + torch.randn(6, dtype=torch.double)
        rotation, offset = fit_drift(before, after)['unitary']
        torch.testing.assert_close(rotation.T @ rotation, torch.eye(6, dtype=torch.double))
        torch.testing.assert_close(after @ rotation + offset, before)

    def test_shift_recovers_a_translation_and_no_drift_is_identity(self):
        before = torch.randn(100, 5, dtype=torch.double)
        shift = torch.randn(5, dtype=torch.double)
        _, offset = fit_drift(before, before + shift)['shift']
        torch.testing.assert_close(offset, -shift)
        for mode, (rotation, offset) in fit_drift(before, before).items():
            if rotation is not None:
                torch.testing.assert_close(rotation, torch.eye(5, dtype=torch.double), atol=1e-8, rtol=0)
            torch.testing.assert_close(offset, torch.zeros(5, dtype=torch.double), atol=1e-8, rtol=0)


class HeisenbergHeadsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)

    def make(self):
        heads = HeisenbergHeads(num_classes=6, dim=4).double()
        head = torch.nn.Linear(4, 6).double()
        return heads, head

    def test_compensation_preserves_old_logits_under_rigid_drift(self):
        heads, head = self.make()
        heads.add_classes([0, 1, 2], head)
        old_features = torch.randn(300, 4, dtype=torch.double)
        drift = orthogonal(4)
        shift = torch.randn(4, dtype=torch.double)
        heads.compensate(old_features, old_features @ drift + shift)
        expected = head(old_features)[:, :3]
        drifted = old_features @ drift + shift
        logits = heads.head_logits(head(drifted), drifted)
        torch.testing.assert_close(logits['unitary'][:, :3], expected)
        torch.testing.assert_close(logits['affine'][:, :3], expected, atol=1e-2, rtol=0)
        self.assertFalse(torch.allclose(logits['shift'][:, :3], expected, atol=1e-2))
        # Unseen classes keep the live classifier logits.
        torch.testing.assert_close(logits['unitary'][:, 3:], head(drifted)[:, 3:])

    @unittest.skipUnless(torch.cuda.is_available(), 'needs CUDA')
    def test_compensate_accepts_cpu_features_for_cuda_buffers(self):
        # engine.collect_head_features returns CPU tensors (Kaggle regression).
        heads = HeisenbergHeads(num_classes=6, dim=4).cuda()
        heads.add_classes([0, 1], torch.nn.Linear(4, 6).cuda())
        features = torch.randn(50, 4)
        stats = heads.compensate(features, features @ orthogonal(4).float())
        self.assertLess(stats['residual_unitary'], 1e-4)

    def test_new_classes_enter_with_trained_rows_and_compose_over_tasks(self):
        heads, head = self.make()
        heads.add_classes([0, 1], head)
        features = torch.randn(200, 4, dtype=torch.double)
        first, second = orthogonal(4), orthogonal(4)
        heads.compensate(features, features @ first)
        heads.add_classes([2, 3], head)
        torch.testing.assert_close(heads.weight[2, 2:4], head.weight[2:4].detach())
        heads.compensate(features @ first, features @ first @ second)
        final = features @ first @ second
        logits = heads.head_logits(head(final), final)['unitary']
        torch.testing.assert_close(logits[:, :2], head(features)[:, :2])
        torch.testing.assert_close(logits[:, 2:4], head(features @ first)[:, 2:4])
        with self.assertRaises(ValueError):
            heads.add_classes([3], head)

    def test_matrix_metrics(self):
        matrix = np.array([[90.0, 84.0, 80.0], [0.0, 88.0, 86.0], [0.0, 0.0, 91.0]])
        metrics = matrix_metrics(matrix, 3)
        self.assertAlmostEqual(metrics['final_avg_acc'], (80 + 86 + 91) / 3)
        self.assertAlmostEqual(metrics['forgetting'], ((90 - 80) + (88 - 86)) / 2)
        self.assertAlmostEqual(metrics['backward_transfer'], ((80 - 90) + (86 - 88)) / 2)


class DriftIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_engine(self, drift):
        torch.manual_seed(10961)
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        args = parser.parse_args(['--drift_heads'] if drift else [])
        args.num_tasks, args.nb_classes, args.epochs = 2, 10, 1
        args.batch_size, args.num_workers, args.pin_mem = 5, 0, False
        args.distributed, args.world_size, args.print_freq = False, 1, 100
        args.top_k, args.length, args.size, args.lr = 2, 2, 4, 0.01
        model = VisionTransformer(
            img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3,
            num_classes=10, prompt_length=2, prompt_pool=True, prompt_key=True,
            pool_size=4, top_k=2, head_type='prompt', batchwise_prompt=True)
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(args.freeze)):
                parameter.requires_grad_(False)
        if drift:
            model.drift_heads = HeisenbergHeads(10, 12, ridge=args.drift_ridge)
        original = VisionTransformer(img_size=16, patch_size=8, embed_dim=12,
                                     depth=1, num_heads=3, num_classes=10)
        original.requires_grad_(False)
        masks = [list(range(5)), list(range(5, 10))]
        full_train = TensorDataset(torch.randn(40, 3, 16, 16), torch.arange(10).repeat(4))
        full_val = TensorDataset(torch.randn(20, 3, 16, 16), torch.arange(10).repeat(2))
        full_train.transform = full_val.transform = None
        loaders = []
        for labels in masks:
            train = Subset(full_train, [i for i in range(40) if i % 10 in labels])
            val = Subset(full_val, [i for i in range(20) if i % 10 in labels])
            loaders.append({'train': DataLoader(train, batch_size=5, shuffle=True),
                            'val': DataLoader(val, batch_size=5)})
        parent = Path(__file__).resolve().parent
        output = parent / ('smoke-' + uuid.uuid4().hex)
        output.mkdir()
        args.output_dir = str(output)
        try:
            summary = train_and_evaluate(
                model, model, original, torch.nn.CrossEntropyLoss(), loaders,
                create_optimizer(args, model), None, torch.device('cpu'), masks, args)
            state = torch.load(output / 'checkpoint/task2_checkpoint.pth', **FULL_PICKLE)
        finally:
            shutil.rmtree(output)
        return summary, model, state

    def test_training_is_unchanged_and_all_heads_are_reported(self):
        baseline, baseline_model, _ = self.run_engine(drift=False)
        summary, model, state = self.run_engine(drift=True)
        for (name, expected), actual in zip(baseline_model.state_dict().items(),
                                            model.state_dict().values()):
            self.assertTrue(torch.equal(expected, actual), name)
        self.assertEqual(summary['accuracy_matrix'], baseline['accuracy_matrix'])
        self.assertNotIn('drift_heads', baseline)
        heads = summary['drift_heads']
        self.assertEqual(set(heads), {'none', 'shift', 'affine', 'unitary'})
        self.assertEqual(heads['none']['final_avg_acc'], summary['final_avg_acc'])
        # Task 1 has no old classes, so every head equals the live classifier there.
        for name in ('shift', 'affine', 'unitary'):
            self.assertEqual(heads[name]['accuracy_matrix'][0][0], summary['accuracy_matrix'][0][0])
        diagnostics = summary['drift_diagnostics']
        self.assertEqual([entry['task'] for entry in diagnostics], [2])
        self.assertGreater(diagnostics[0]['drift'], 0)
        self.assertLessEqual(diagnostics[0]['residual_affine'], diagnostics[0]['drift'] + 1e-9)
        self.assertEqual(int(state['model']['drift_heads.seen'].sum()), 10)


if __name__ == '__main__':
    unittest.main()
