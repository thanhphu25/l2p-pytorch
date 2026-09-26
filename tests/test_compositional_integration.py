"""CPU smoke test of the actual ViT + two-task engine, without dataset downloads."""
import argparse
import inspect
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid

import torch
from torch.utils.data import DataLoader, TensorDataset
from timm.optim import create_optimizer

from compositional_prompt import validate_compositional_args
from configs.cifar100_l2p import get_args_parser
from engine import train_and_evaluate
from vision_transformer import VisionTransformer

# Checkpoints pickle argparse args; torch<1.13 has no weights_only argument.
FULL_PICKLE = ({'weights_only': False}
               if 'weights_only' in inspect.signature(torch.load).parameters else {})


class CompositionalIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10961)
        torch.set_num_threads(1)

    def args(self, router='qsd_comp'):
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        args = parser.parse_args(['--prompt_router', router, '--no_batchwise_prompt'])
        args.num_tasks = 2
        args.nb_classes = 10
        args.epochs = 1
        args.batch_size = 5
        args.distributed = False
        args.world_size = 1
        args.output_dir = ''
        args.print_freq = 100
        args.comp_candidates_per_class = 2
        args.comp_prototypes_per_class = 2
        args.comp_components_per_task = 2
        args.top_k = 2
        args.length = 2
        args.qsd_state_dim = 4
        args.qsd_rank = 2
        args.lr = 0.001
        return args

    def model(self, router='qsd_comp'):
        return VisionTransformer(
            img_size=16, patch_size=8, embed_dim=12, depth=1, num_heads=3,
            num_classes=10, prompt_length=2, prompt_pool=True, prompt_key=True,
            pool_size=10, top_k=2, head_type='prompt', prompt_router=router,
            qsd_state_dim=4, qsd_rank=2, comp_num_tasks=2,
            comp_components_per_task=2, comp_prototypes_per_class=2)

    def test_vit_init_keeps_projection_and_eval_ignores_task_id(self):
        model = self.model().eval()
        torch.testing.assert_close(model.prompt.projection.T @ model.prompt.projection,
                                   torch.eye(4), atol=1e-6, rtol=1e-5)
        model.prompt.begin_task(1)
        inputs, cls = torch.randn(2, 3, 16, 16), torch.randn(2, 12)
        patches = torch.randn(2, 4, 12)
        expected = model(inputs, cls_features=cls, prompt_query_tokens=patches, task_id=0)
        actual = model(inputs, cls_features=cls, prompt_query_tokens=patches, task_id=999)
        torch.testing.assert_close(actual['logits'], expected['logits'])
        self.assertEqual(actual['x'].shape, (2, 9, 12))

    def test_two_task_engine_and_checkpoint_for_quantum_and_cosine(self):
        for router in ('qsd_comp', 'cosine_comp'):
            with self.subTest(router=router):
                self.run_two_tasks(router)

    def run_two_tasks(self, router):
        args = self.args(router)
        validate_compositional_args(args)
        model = self.model(router)
        original = VisionTransformer(img_size=16, patch_size=8, embed_dim=12,
                                     depth=1, num_heads=3, num_classes=10)
        original.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith(tuple(args.freeze)):
                parameter.requires_grad_(False)
        backbone_before = {name: p.clone() for name, p in model.named_parameters()
                           if name.startswith(tuple(args.freeze))}
        bank_snapshots = []
        def check_freezing(module, inputs):
            if int(module.active_tasks) == 1:
                bank_snapshots[:] = [module.prompts[0].detach().clone(),
                                     module.factors[0].detach().clone()]
            elif bank_snapshots:
                self.assertTrue(torch.equal(module.prompts[0], bank_snapshots[0]))
                self.assertTrue(torch.equal(module.factors[0], bank_snapshots[1]))
        handle = model.prompt.register_forward_pre_hook(check_freezing)
        loaders = []
        masks = [list(range(5)), list(range(5, 10))]
        for labels in masks:
            train = TensorDataset(torch.randn(10, 3, 16, 16), torch.tensor(labels * 2))
            val = TensorDataset(torch.randn(5, 3, 16, 16), torch.tensor(labels))
            loaders.append({'train': DataLoader(train, batch_size=5),
                            'val': DataLoader(val, batch_size=5)})
        # Default permissions also work on Windows sandbox ACLs.
        parent = Path(__file__).resolve().parent
        output = parent / ('smoke-' + uuid.uuid4().hex)
        output.mkdir()
        args.output_dir = str(output)
        try:
            optimizer = create_optimizer(args, model)
            summary = train_and_evaluate(
                model, model, original, torch.nn.CrossEntropyLoss(), loaders,
                optimizer, None, torch.device('cpu'), masks, args)
            self.assertEqual(summary['tasks_completed'], 2)
            self.assertEqual(summary['compositional_config']['active_components'], 4)
            self.assertEqual(summary['final_router_metrics_source'], 'evaluation')
            self.assertEqual(summary['final_router_metrics']['MemCount'], 20)
            self.assertGreater(summary['final_router_metrics']['CompKL'], 0)
            json.dumps(summary)
            state = torch.load(output / 'checkpoint/task2_checkpoint.pth', **FULL_PICKLE)
            restored = self.model(router)
            restored.load_state_dict(state['model'])
            restored.eval()
            model.eval()
            inputs = loaders[1]['val'].dataset.tensors[0]
            frozen = original(inputs)
            kwargs = dict(cls_features=frozen['pre_logits'], prompt_query_tokens=frozen['x'][:, 1:])
            torch.testing.assert_close(restored(inputs, **kwargs)['logits'], model(inputs, **kwargs)['logits'])
            self.assertEqual([p.requires_grad for p in restored.prompt.prompts], [False, True])
            self.assertEqual(restored.prompt.memory_counts.sum().item(), 20)
            for name, parameter in model.named_parameters():
                if name in backbone_before:
                    self.assertTrue(torch.equal(parameter, backbone_before[name]))
        finally:
            handle.remove()
            if output.resolve().parent != parent:
                raise RuntimeError('Temporary test output escaped its parent directory')
            shutil.rmtree(output)

    def test_cli_defaults_preserve_baseline_and_reject_incompatible_modes(self):
        parser = argparse.ArgumentParser()
        get_args_parser(parser)
        default = parser.parse_args([])
        self.assertEqual(default.prompt_router, 'cosine')
        self.assertTrue(default.batchwise_prompt)
        args = self.args()
        validate_compositional_args(args)
        for name in ('batchwise_prompt', 'use_prompt_mask', 'task_inc', 'shared_prompt_pool', 'distributed'):
            with self.subTest(name=name):
                setattr(args, name, True)
                with self.assertRaises(ValueError):
                    validate_compositional_args(args)
                setattr(args, name, False)


if __name__ == '__main__':
    unittest.main()
