"""Checks for density_head.py and its engine integration. Run: python -m pytest tests -q (or python tests/test_density_head.py)"""
import io
import os
import sys
import types
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from density_head import DensityHeads, HeadMetrics  # noqa: E402

D = 24


def make_args(**kw):
    args = dict(density_sources=['frozen', 'prompted'], density_rank=8, density_ranks=[2, 4],
                density_eps=1e-4, density_fusion_weight=1.0, density_fusion_weight2=None, density_lda_shrink=0.1)
    args.update(kw)
    return Namespace(**args)


def clustered(classes, n, seed=0, spread=0.3):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(max(classes) + 1, D, generator=g) * 3
    X, y = [], []
    for c in classes:
        X.append(centers[c] + spread * torch.randn(n, D, generator=g))
        y.append(torch.full((n,), c))
    return torch.cat(X), torch.cat(y)


def built_heads(classes=(0, 1, 2, 3, 4, 5), n=12, **kw):
    heads = DensityHeads(make_args(**kw), D)
    X, y = clustered(list(classes), n)
    heads.add_task({'frozen': X, 'prompted': X * 2 + 1}, y)
    return heads, X, y


def test_pgm_complete():
    heads, X, _ = built_heads()
    xn = torch.nn.functional.normalize(torch.randn(50, D, dtype=torch.float64), dim=1)
    for bank in heads.banks.values():
        for rank in [bank.rank] + heads.ranks:
            p = bank.pgm_probs(xn, rank)
            assert (p > 0).all()
            assert torch.allclose(p.sum(1), torch.ones(50, dtype=torch.float64), atol=1e-8), p.sum(1)


def test_rank_larger_than_samples():
    heads = DensityHeads(make_args(density_rank=16), D)
    X, y = clustered([0, 1, 2], 5)  # 5 samples per class < rank 16
    heads.add_task({'frozen': X, 'prompted': X}, y)
    bank = heads.banks['frozen']
    assert torch.allclose(bank.lam.sum(1), torch.ones(3))
    assert (bank.lam[:, 5:] == 0).all()
    xn = torch.nn.functional.normalize(torch.randn(7, D, dtype=torch.float64), dim=1)
    assert torch.allclose(bank.pgm_probs(xn, 16).sum(1), torch.ones(7, dtype=torch.float64), atol=1e-8)


def test_unseen_classes_never_predicted():
    heads, X, y = built_heads(classes=(0, 1, 2))
    logits = torch.zeros(X.shape[0], 10)
    logits[:, 7] = 100.0  # unseen class dominates raw logits
    preds = heads.predict({'frozen': X, 'prompted': X * 2 + 1}, logits)
    assert set(preds) == set(heads.head_names())
    assert (preds['linear'] == 7).all()
    for name, p in preds.items():
        if name != 'linear':
            assert set(p.tolist()) <= {0, 1, 2}, name


def test_heads_separate_clusters():
    heads, X, y = built_heads()
    Xt, yt = clustered([0, 1, 2, 3, 4, 5], 10, seed=0)
    logits = torch.randn(Xt.shape[0], 6)
    preds = heads.predict({'frozen': Xt, 'prompted': Xt * 2 + 1}, logits)
    for name in ['frozen_pgm', 'frozen_pgm_fusion', 'frozen_ncm_whitened', 'frozen_lda', 'prompted_pgm_r4',
                 'dual_pgm_fusion', 'dual_pgm_fusion_r2']:
        acc = (preds[name] == yt).float().mean().item()
        assert acc > 0.9, (name, acc)


def test_incremental_add_and_cache():
    heads, _, _ = built_heads(classes=(0, 1, 2))
    p_before = heads.banks['frozen'].measurement(8, 'cpu')['M']
    X, y = clustered([3, 4], 12, seed=1)
    heads.add_task({'frozen': X, 'prompted': X}, y)
    assert heads.classes == [0, 1, 2, 3, 4]
    assert p_before == 3 and heads.banks['frozen'].measurement(8, 'cpu')['M'] == 5


def test_checkpoint_round_trip():
    heads, X, _ = built_heads()
    logits = torch.randn(X.shape[0], 6)
    feats = {'frozen': X, 'prompted': X * 2 + 1}
    before = heads.predict(feats, logits)
    buf = io.BytesIO()
    torch.save({'density_bank': heads.state_dict()}, buf)
    buf.seek(0)
    loaded = DensityHeads(make_args(density_ranks=[], density_fusion_weight=3.0), D)
    loaded.load_state_dict(torch.load(buf)['density_bank'])
    after = loaded.predict(feats, logits)
    assert loaded.ranks == heads.ranks and loaded.weight == heads.weight
    for k in before:
        assert torch.equal(before[k], after[k]), k


def test_head_metrics():
    m = HeadMetrics(3)
    m.update('h', 0, 0, 90.0)
    s0 = m.end_task(0)
    m.update('h', 0, 1, 80.0)
    m.update('h', 1, 1, 70.0)
    s1 = m.end_task(1)
    assert s0['h']['acc'] == 90.0 and s1['h']['acc'] == 75.0
    assert s1['h']['incremental_acc'] == 82.5 and s1['h']['forgetting'] == 10.0


def test_offline_eval_keeps_args():
    heads, X, _ = built_heads()
    buf = io.BytesIO()
    torch.save({'density_bank': heads.state_dict()}, buf)
    buf.seek(0)
    # rank 8 and 32 are not below the stored rank 8, so they are dropped
    loaded = DensityHeads(make_args(density_rank=32, density_ranks=[1, 2, 8, 32], density_fusion_weight=0.5,
                                    density_fusion_weight2=2.0), D)
    loaded.load_state_dict(torch.load(buf)['density_bank'], use_saved_config=False)
    assert loaded.rank == 8 and loaded.ranks == [1, 2]
    assert loaded.weight == 0.5 and loaded.weight2 == 2.0
    preds = loaded.predict({'frozen': X, 'prompted': X * 2 + 1}, torch.randn(X.shape[0], 6))
    assert set(preds) == set(loaded.head_names())
    assert 'frozen_pgm_r1' in preds and 'dual_pgm_fusion_r1' in preds


def test_dual_fusion_weights():
    heads, X, _ = built_heads()
    Xt, _ = clustered([0, 1, 2, 3, 4, 5], 10, seed=3, spread=3.0)  # overlapping clusters, so weights matter
    feats = {'frozen': Xt, 'prompted': Xt * 2 + 1}
    logits = torch.randn(Xt.shape[0], 6)
    # w2 = 0 leaves only the frozen term, i.e. frozen_pgm_fusion
    heads.weight2 = 0.0
    preds = heads.predict(feats, logits)
    assert torch.equal(preds['dual_pgm_fusion'], preds['frozen_pgm_fusion'])
    assert torch.equal(preds['dual_pgm_fusion_r4'], preds['frozen_pgm_fusion_r4'])
    # single source: no dual heads
    single = DensityHeads(make_args(density_sources=['prompted']), D)
    single.add_task({'prompted': X}, torch.arange(6).repeat_interleave(12))
    assert not any(h.startswith('dual') for h in single.predict({'prompted': Xt}, logits))


def test_train_eval_copy_keeps_train_transform():
    import datasets as cl_datasets

    class Toy(torch.utils.data.Dataset):
        def __init__(self):
            self.transform = 'train'

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return self.transform, i

    base = Toy()
    subsets = [torch.utils.data.Subset(base, [0, 1]), torch.utils.data.Subset(base, [2, 3])]
    cache = {}
    copies = [cl_datasets._with_transform(s, 'eval', cache) for s in subsets]
    assert base.transform == 'train'
    assert [c[0] for c in copies] == [('eval', 0), ('eval', 2)]
    assert copies[0].dataset is copies[1].dataset  # one copy of the underlying dataset for every task


def _import_engine():
    # engine imports timm only for accuracy/create_optimizer; stub it when timm is not installed
    try:
        import timm  # noqa: F401
    except ImportError:
        timm = types.ModuleType('timm')
        tu, to = types.ModuleType('timm.utils'), types.ModuleType('timm.optim')

        def accuracy(output, target, topk=(1,)):
            maxk = min(max(topk), output.size(1))
            pred = output.topk(maxk, 1, True, True)[1].t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))
            return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / target.size(0) for k in topk]
        tu.accuracy = accuracy
        to.create_optimizer = None
        sys.modules.update({'timm': timm, 'timm.utils': tu, 'timm.optim': to})
    import engine
    return engine


class FakeModel(torch.nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.proj = torch.nn.Linear(D, D)
        self.head = torch.nn.Linear(D, n_classes)

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        f = self.proj(x)
        return {'pre_logits': f, 'logits': self.head(f)}


class FakeOriginal(torch.nn.Module):
    def forward(self, x):
        return {'pre_logits': x}


def test_engine_consolidate_and_evaluate():
    engine = _import_engine()
    torch.manual_seed(0)
    X, y = clustered([0, 1, 2], 10)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=8)
    model, original = FakeModel(), FakeOriginal()
    heads = DensityHeads(make_args(), D)

    rng_before = (torch.get_rng_state(), np.random.get_state()[1].copy())
    engine.consolidate_density_heads(model, original, loader, 'cpu', 0, heads)
    assert torch.equal(torch.get_rng_state(), rng_before[0])
    assert np.array_equal(np.random.get_state()[1], rng_before[1])
    assert heads.classes == [0, 1, 2]

    args = Namespace(print_freq=100, task_inc=False, output_dir='')
    stats = engine.evaluate(model, original, loader, 'cpu', task_id=0, args=args, density=heads)
    assert abs(stats['density']['linear'] - stats['Acc@1']) < 1e-6
    assert stats['density']['frozen_pgm'] > 90.0

    no_density = engine.evaluate(model, original, loader, 'cpu', task_id=0, args=args)
    assert 'density' not in no_density and no_density['Acc@1'] == stats['Acc@1']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
