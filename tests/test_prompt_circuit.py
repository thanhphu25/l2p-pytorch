import argparse
import inspect
from pathlib import Path
import shutil
import unittest
import uuid

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from timm.optim import create_optimizer

from configs.cifar100_l2p import get_args_parser
from engine import train_and_evaluate
from prompt_circuit import PromptCircuit
from vision_transformer import Attention, VisionTransformer

# Checkpoints pickle argparse args; torch<1.13 (Kaggle pin 1.12.1) has no weights_only.
FULL_PICKLE = ({'weights_only': False}
               if 'weights_only' in inspect.signature(torch.load).parameters else {})


def perturb(circuit, scale=0.5):
    with torch.no_grad():
        circuit.left.normal_(0, scale)


class PromptCircuitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)

    def test_every_mode_starts_as_the_retrieved_prompt_on_each_layer(self):
        prompt = torch.randn(3, 4, 8)
        for mode in ('unitary', 'linear', 'shared'):
            prefixes, _ = PromptCircuit(8, [1, 3], mode=mode, rank=2)(prompt)
            self.assertEqual(sorted(prefixes), [1, 3])
            for prefix in prefixes.values():
                torch.testing.assert_close(prefix, torch.stack([prompt, prompt], 1))

    def test_unitary_layers_preserve_norms_and_pool_geometry(self):
        circuit = PromptCircuit(8, [1, 2, 3], mode='unitary', rank=2).double()
        perturb(circuit)
        maps = circuit.layer_maps()
        identity = torch.eye(8, dtype=torch.double).expand_as(maps)
        torch.testing.assert_close(maps @ maps.transpose(-1, -2), identity)
        pool = torch.randn(5, 2, 8, dtype=torch.double)
        prompt = pool[:2].reshape(1, 4, 8)
        prefixes, diagnostics = circuit(prompt, pool=pool)
        self.assertFalse(torch.allclose(prefixes[3][:, 0], prompt))  # it did evolve
        torch.testing.assert_close(prefixes[3][:, 0].norm(dim=-1), prompt.norm(dim=-1))
        self.assertAlmostEqual(diagnostics['circuit_norm_ratio'].item(), 1.0, places=10)
        self.assertLess(diagnostics['circuit_gram_drift'].item(), 1e-10)

    def test_low_rank_exponential_equals_dense_matrix_exp(self):
        circuit = PromptCircuit(16, [1, 2], mode='unitary', rank=3).double()
        perturb(circuit, scale=0.8)
        generator = circuit.left @ circuit.right.transpose(-1, -2)
        dense = torch.stack([torch.matrix_exp(a) for a in
                             (generator - generator.transpose(-1, -2)).flatten(0, 1)])
        torch.testing.assert_close(circuit.layer_maps().flatten(0, 1), dense)

    def test_linear_control_has_same_parameters_but_distorts_geometry(self):
        unitary = PromptCircuit(8, [1, 2], mode='unitary', rank=2)
        linear = PromptCircuit(8, [1, 2], mode='linear', rank=2)
        self.assertEqual(sum(p.numel() for p in unitary.parameters()),
                         sum(p.numel() for p in linear.parameters()))
        perturb(linear)
        pool = torch.randn(5, 2, 8)
        _, diagnostics = linear(pool[:2].reshape(1, 4, 8), pool=pool)
        self.assertGreater(diagnostics['circuit_gram_drift'].item(), 0.05)

    def test_independent_mode_selects_per_layer_prompts_by_l2p_index(self):
        circuit = PromptCircuit(8, [2, 4], mode='independent', pool_size=5, length=3)
        index = torch.tensor([[0, 4], [2, 2]])
        prefixes, diagnostics = circuit(torch.randn(2, 6, 8), prompt_idx=index)
        self.assertEqual(prefixes[4].shape, (2, 2, 6, 8))
        torch.testing.assert_close(prefixes[4][1, 0, :3], circuit.pool[1, 0, 2])
        torch.testing.assert_close(prefixes[2][0, 1, 3:], circuit.pool[0, 1, 4])
        self.assertEqual(diagnostics, {})

    def test_attention_prefix_matches_explicit_concatenation(self):
        attention = Attention(8, num_heads=2).eval()
        x, prefix = torch.randn(2, 5, 8), torch.randn(2, 2, 3, 8)
        q, k, v = attention.qkv(x).reshape(2, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
        heads = lambda t: t.reshape(2, 3, 2, 4).transpose(1, 2)
        k = torch.cat((heads(prefix[:, 0]), k), 2)
        v = torch.cat((heads(prefix[:, 1]), v), 2)
        expected = ((q @ k.transpose(-1, -2)) * attention.scale).softmax(-1) @ v
        expected = attention.proj(expected.transpose(1, 2).reshape(2, 5, 8))
        torch.testing.assert_close(attention(x, prefix=prefix), expected)
        self.assertEqual(attention(x).shape, (2, 5, 8))


def tiny_vit(mode='none', **kwargs):
    return VisionTransformer(
        img_size=16, patch_size=8, embed_dim=12, depth=3, num_heads=3,
        num_classes=10, prompt_length=2, prompt_pool=True, prompt_key=True,
        pool_size=4, top_k=2, head_type='prompt', batchwise_prompt=True,
        circuit_mode=mode, circuit_layers=(1, 2), circuit_rank=2, **kwargs)


class CircuitViTTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)
        torch.set_num_threads(1)

    def test_none_mode_is_plain_l2p_and_circuit_receives_ce_gradient(self):
        torch.manual_seed(0)
        baseline = tiny_vit('none').eval()
        torch.manual_seed(0)
        circuit = tiny_vit('unitary').eval()
        missing = circuit.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(set(missing.missing_keys), {'prompt_circuit.left', 'prompt_circuit.right'})
        images, cls = torch.randn(4, 3, 16, 16), torch.randn(4, 12)
        plain = baseline(images, cls_features=cls)['logits']
        with torch.no_grad():
            circuit.prompt_circuit.left.zero_()
        self.assertFalse(torch.allclose(circuit(images, cls_features=cls)['logits'], plain))
        circuit.train()
        output = circuit(images, cls_features=cls, train=True)
        F.cross_entropy(output['logits'], torch.arange(4)).backward()
        self.assertGreater(circuit.prompt_circuit.left.grad.abs().sum().item(), 0)
        self.assertIn('circuit_norm_ratio', output)

    def test_invalid_layers_are_rejected(self):
        with self.assertRaises(ValueError):
            VisionTransformer(img_size=16, patch_size=8, embed_dim=12, depth=2, num_heads=3,
                              num_classes=10, prompt_length=2, prompt_pool=True, pool_size=4,
                              top_k=2, circuit_mode='unitary', circuit_layers=(1, 2))


class CircuitEngineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_mode(self, mode):
        torch.manual_seed(10961)
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        args = parser.parse_args(['--circuit_mode', mode, '--circuit_layers', '1', '2',
                                  '--circuit_rank', '2'])
        args.num_tasks, args.nb_classes, args.epochs = 2, 10, 1
        args.batch_size, args.distributed, args.world_size, args.print_freq = 5, False, 1, 100
        args.top_k, args.length, args.size, args.lr = 2, 2, 4, 0.01
        model = tiny_vit(mode)
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(args.freeze)):
                parameter.requires_grad_(False)
        frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
        original = VisionTransformer(img_size=16, patch_size=8, embed_dim=12, depth=3,
                                     num_heads=3, num_classes=10)
        original.requires_grad_(False)
        masks = [list(range(5)), list(range(5, 10))]
        loaders = []
        for labels in masks:
            train = TensorDataset(torch.randn(10, 3, 16, 16), torch.tensor(labels * 2))
            val = TensorDataset(torch.randn(5, 3, 16, 16), torch.tensor(labels))
            loaders.append({'train': DataLoader(train, batch_size=5),
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
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(parameter, frozen[name]), name)
        return summary, model, state

    def test_two_tasks_for_every_mode(self):
        for mode in ('unitary', 'linear', 'shared', 'independent'):
            with self.subTest(mode=mode):
                summary, model, state = self.run_mode(mode)
                self.assertEqual(summary['tasks_completed'], 2)
                self.assertEqual(summary['circuit_config']['mode'], mode)
                self.assertIn(mode, summary['method'])
                restored = tiny_vit(mode)
                restored.load_state_dict(state['model'])
                metrics = summary['final_router_metrics']
                if mode == 'unitary':
                    self.assertAlmostEqual(metrics['CircNorm'], 1.0, places=4)
                    self.assertLess(metrics['CircDrift'], 1e-4)
                    self.assertGreater(model.prompt_circuit.left.abs().sum().item(), 0)
                if mode == 'linear':
                    self.assertGreater(metrics['CircDrift'], 0)

    def test_cli_default_is_plain_l2p(self):
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        args = parser.parse_args([])
        self.assertEqual(args.circuit_mode, 'none')
        self.assertEqual(args.circuit_layers, [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
