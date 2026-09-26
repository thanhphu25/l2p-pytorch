import argparse
import inspect
import io
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from timm.optim import create_optimizer

from configs.cifar100_l2p import get_args_parser
from density_head import DensityClassBank, DensityHeads
from engine import train_and_evaluate
from result_summary import build_results_summary, matrix_metrics
from vision_transformer import VisionTransformer

# Checkpoints pickle argparse args; torch<1.13 (Kaggle pin 1.12.1) has no weights_only.
FULL_PICKLE = ({'weights_only': False}
               if 'weights_only' in inspect.signature(torch.load).parameters else {})


class DensityClassBankTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)

    def test_states_are_trace_one_orthonormal_and_immutable(self):
        bank = DensityClassBank(num_classes=4, dim=8, rank=3)
        bank.add_class(2, torch.randn(20, 8))
        vectors = bank.vectors[2]
        torch.testing.assert_close(vectors.T @ vectors, torch.eye(3), atol=1e-5, rtol=0)
        self.assertAlmostEqual(bank.values[2].sum().item(), 1.0, places=5)
        self.assertTrue(torch.all(bank.values[2][:-1] >= bank.values[2][1:]))
        with self.assertRaises(ValueError):
            bank.add_class(2, torch.randn(5, 8))

    def test_rank_is_capped_by_sample_count(self):
        bank = DensityClassBank(num_classes=2, dim=8, rank=5)
        bank.add_class(0, torch.randn(2, 8))
        self.assertEqual(int((bank.values[0] > 0).sum()), 2)
        torch.testing.assert_close(bank.vectors[0, :, 2:], torch.zeros(8, 3))

    def test_pgm_is_a_complete_povm(self):
        bank = DensityClassBank(num_classes=5, dim=6, rank=2, eps=1e-3).double()
        for label in range(4):
            bank.add_class(label, torch.randn(10, 6, dtype=torch.double))
        root = bank._pgm_root()
        states = [bank.vectors[c] @ torch.diag(bank.values[c]) @ bank.vectors[c].T for c in range(4)]
        effects = [root @ ((s + bank.eps * torch.eye(6, dtype=torch.double)) / 4) @ root for s in states]
        torch.testing.assert_close(sum(effects), torch.eye(6, dtype=torch.double))
        for effect in effects:
            self.assertGreater(torch.linalg.eigvalsh(effect).min().item(), -1e-12)
        # The closed form used by scores() matches <x|E_c|x> and sums to one.
        x = F.normalize(torch.randn(3, 6, dtype=torch.double), dim=-1)
        explicit = torch.stack([torch.einsum('bi,ij,bj->b', x, e, x) for e in effects], -1)
        torch.testing.assert_close(explicit.sum(-1), torch.ones(3, dtype=torch.double))
        pgm = bank.scores(x)['pgm']
        torch.testing.assert_close(pgm[:, :4].exp(), explicit)
        self.assertTrue(torch.isinf(pgm[:, 4]).all())

    def test_rank_one_fidelity_ranks_like_squared_cosine(self):
        bank = DensityClassBank(num_classes=3, dim=5, rank=1)
        samples = torch.randn(3, 5)
        for label in range(3):
            bank.add_class(label, samples[label:label + 1])
        x = torch.randn(7, 5)
        cosine = F.normalize(x, dim=-1) @ F.normalize(samples, dim=-1).T
        scores = bank.scores(x)
        self.assertTrue(torch.equal(scores['fidelity'].argmax(-1), cosine.square().argmax(-1)))
        torch.testing.assert_close(scores['ncm'], cosine, atol=1e-5, rtol=0)

    def test_orthogonal_class_states_are_discriminated_perfectly(self):
        bank = DensityClassBank(num_classes=3, dim=9, rank=3, eps=1e-6)
        basis = torch.linalg.qr(torch.randn(9, 9))[0]
        for label in range(3):
            span = basis[:, 3 * label:3 * label + 3]
            bank.add_class(label, torch.randn(12, 3) @ span.T)
        test = torch.cat([torch.randn(4, 3) @ basis[:, 3 * c:3 * c + 3].T for c in range(3)])
        labels = torch.arange(3).repeat_interleave(4)
        # Zero-mean classes: only the density read-outs see the subspaces.
        for name, value in bank.scores(test, readouts=('fidelity', 'pgm')).items():
            self.assertTrue(torch.equal(value.argmax(-1), labels), name)
        born = bank.scores(test)['pgm'].exp()
        self.assertGreater(born[torch.arange(12), labels].min().item(), 0.999)

    def test_measurement_cache_follows_class_set_and_checkpoints(self):
        bank = DensityClassBank(num_classes=3, dim=4, rank=2)
        bank.add_class(0, torch.randn(6, 4))
        bank.add_class(1, torch.randn(6, 4))
        first = bank.scores(torch.randn(2, 4))['pgm']
        other = DensityClassBank(num_classes=3, dim=4, rank=2)
        other.add_class(0, torch.randn(6, 4))
        other.add_class(1, torch.randn(6, 4))
        x = torch.randn(2, 4)
        bank.scores(x)  # warm the cache with the old states
        stream = io.BytesIO()
        torch.save(other.state_dict(), stream)
        stream.seek(0)
        bank.load_state_dict(torch.load(stream))
        torch.testing.assert_close(bank.scores(x)['pgm'], other.scores(x)['pgm'])
        self.assertEqual(first.shape, (2, 3))

    def test_heads_mask_unseen_classes_including_fusion(self):
        heads = DensityHeads(['frozen'], num_classes=6, dim=4, rank=2)
        heads.banks['frozen'].add_class(0, torch.randn(5, 4))
        heads.banks['frozen'].add_class(3, torch.randn(5, 4))
        linear = torch.zeros(2, 6)
        linear[:, 5] = 100.0  # unseen class with a huge logit must never win
        logits = heads.head_logits(linear, {'frozen': torch.randn(2, 4)})
        self.assertEqual(set(logits), {'frozen_ncm', 'frozen_fidelity', 'frozen_pgm', 'frozen_fusion'})
        for value in logits.values():
            self.assertTrue(set(value.argmax(-1).tolist()) <= {0, 3})


class DensitySummaryTest(unittest.TestCase):
    def test_matrix_metrics_match_engine_definitions(self):
        matrix = np.array([[90.0, 84.0, 80.0], [0.0, 88.0, 86.0], [0.0, 0.0, 91.0]])
        metrics = matrix_metrics(matrix, 3)
        self.assertAlmostEqual(metrics['final_avg_acc'], (80 + 86 + 91) / 3)
        self.assertAlmostEqual(metrics['avg_incremental_acc'], (90 + 86 + (80 + 86 + 91) / 3) / 3)
        self.assertAlmostEqual(metrics['forgetting'], ((90 - 80) + (88 - 86)) / 2)
        self.assertAlmostEqual(metrics['backward_transfer'], ((80 - 90) + (86 - 88)) / 2)


class DensityIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)
        torch.set_num_threads(1)

    def args(self, density):
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        args = parser.parse_args(['--density_head'] if density else [])
        args.num_tasks, args.nb_classes, args.epochs = 2, 10, 1
        args.batch_size, args.num_workers, args.pin_mem = 5, 0, False
        args.distributed, args.world_size, args.print_freq = False, 1, 100
        args.top_k, args.length, args.size, args.lr = 2, 2, 4, 0.001
        args.density_rank = 3
        return args

    def model(self, args):
        model = VisionTransformer(
            img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3,
            num_classes=10, prompt_length=2, prompt_pool=True, prompt_key=True,
            pool_size=4, top_k=2, head_type='prompt', batchwise_prompt=True)
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(args.freeze)):
                parameter.requires_grad_(False)
        if args.density_head:
            model.density_heads = DensityHeads(
                args.density_sources, 10, 12, rank=args.density_rank, eps=args.density_eps)
        return model

    def loaders(self):
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
        return loaders, masks

    def run_engine(self, density):
        torch.manual_seed(10961)
        args = self.args(density)
        model = self.model(args)
        original = VisionTransformer(img_size=16, patch_size=8, embed_dim=12,
                                     depth=1, num_heads=3, num_classes=10)
        original.requires_grad_(False)
        loaders, masks = self.loaders()
        parent = Path(__file__).resolve().parent
        output = parent / ('smoke-' + uuid.uuid4().hex)
        output.mkdir()
        args.output_dir = str(output)
        try:
            optimizer = create_optimizer(args, model)
            summary = train_and_evaluate(
                model, model, original, torch.nn.CrossEntropyLoss(), loaders,
                optimizer, None, torch.device('cpu'), masks, args)
            state = torch.load(output / 'checkpoint/task2_checkpoint.pth', **FULL_PICKLE)
        finally:
            shutil.rmtree(output)
        return summary, model, state

    def test_density_heads_leave_training_unchanged_and_are_checkpointed(self):
        baseline, baseline_model, _ = self.run_engine(density=False)
        summary, model, state = self.run_engine(density=True)
        # Same seed: identical L2P weights and linear-head accuracy matrix.
        for (name, expected), actual in zip(baseline_model.state_dict().items(),
                                            model.state_dict().values()):
            self.assertTrue(torch.equal(expected, actual), name)
        self.assertEqual(summary['accuracy_matrix'], baseline['accuracy_matrix'])
        self.assertNotIn('density_heads', baseline)

        heads = summary['density_heads']
        expected = {'linear'} | {'{}_{}'.format(source, readout)
                                 for source in ('frozen', 'prompted')
                                 for readout in ('ncm', 'fidelity', 'pgm', 'fusion')}
        self.assertEqual(set(heads), expected)
        self.assertEqual(heads['linear']['final_avg_acc'], summary['final_avg_acc'])
        self.assertEqual(len(heads['frozen_pgm']['accuracy_matrix']), 2)
        self.assertEqual(summary['density_config']['rank'], 3)
        self.assertEqual(int(state['model']['density_heads.banks.frozen.valid'].sum()), 10)

        restored = self.model(self.args(density=True))
        restored.load_state_dict(state['model'])
        torch.testing.assert_close(restored.density_heads.banks['prompted'].vectors,
                                   model.density_heads.banks['prompted'].vectors)

    def test_summary_without_heads_is_unchanged(self):
        args = self.args(density=False)
        args.prompt_router = 'cosine'
        summary = build_results_summary(args, [{
            'task': 1, 'avg_acc1': 90.0, 'avg_acc5': 99.0, 'avg_loss': 0.3,
            'forgetting': 0.0, 'backward_transfer': 0.0}], np.array([[90.0]]))
        self.assertNotIn('density_heads', summary)


if __name__ == '__main__':
    unittest.main()
