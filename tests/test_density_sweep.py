"""Checks for density_sweep.py and the feature dump. Run: python -m pytest tests -q"""
import csv
import os
import sys
from argparse import Namespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import density_sweep as sw  # noqa: E402
from density_head import ClassBank, DensityHeads, HeadMetrics  # noqa: E402
from test_density_head import D, FakeModel, FakeOriginal, _import_engine, clustered, make_args  # noqa: E402


def _bank(classes=range(6), n=15, spread=1.5, rank=8):
    X, y = clustered(list(classes), n, spread=spread)
    bank = ClassBank(D, rank)
    for c in classes:
        bank.add_class(c, X[y == c])
    return bank


def test_power_family_alpha1_is_pgm():
    bank = _bank()
    xn = F.normalize(torch.randn(40, D, dtype=torch.float64), dim=1)
    for r in (2, 8):
        lp = sw.pgm_logp(sw.Whitening(bank, r, 1.0, 'cpu'), xn, [1e-4], chunk=16)[1e-4]
        assert torch.allclose(lp.exp(), bank.pgm_probs(xn, r), atol=1e-10)
    # every alpha gives a complete measurement: probabilities sum to one before renormalization
    wh = sw.Whitening(bank, 8, 3.0, 'cpu')
    S_mh = wh.S_mh(1e-4)
    y = xn @ S_mh
    p = (((y @ wh.U_flat).view(40, wh.M, wh.r) ** 2 * wh.lam).sum(-1) + 1e-4 * (y ** 2).sum(-1, keepdim=True)) / wh.M
    assert torch.allclose(p.sum(1), torch.ones(40, dtype=torch.float64), atol=1e-8)


def test_opt_measurement():
    bank = _bank(spread=3.0)
    iters = [0, 1, 5, 30]
    wh, snaps, success = sw.opt_measurements(bank, 4, 1e-4, 1e-9, iters, 'cpu')
    seq = [success[k] for k in range(max(iters) + 1)]
    # success probability never decreases (up to the ridge added to G^2)
    assert all(b >= a - 1e-10 for a, b in zip(seq, seq[1:])), seq
    assert seq[-1] > seq[0]
    xn = F.normalize(torch.randn(30, D, dtype=torch.float64), dim=1)
    for k in iters[1:]:
        Q, K = snaps[k]
        Z = ((xn @ Q) @ wh.U_flat).view(30, wh.M, wh.r)
        s = (torch.einsum('nmr,mrs->nms', Z, K) * Z).sum(-1)
        assert (s >= -1e-12).all() and torch.allclose(s.sum(1), torch.ones(30, dtype=torch.float64), atol=1e-3)
    # iteration 0 is the PGM readout (without its class-independent eps floor)
    lp0 = sw.opt_logp(wh, *snaps[0], xn, chunk=8)
    assert torch.equal(lp0.argmax(1), bank.pgm_probs(xn, 4).argmax(1))


def _loaders(tasks, n_train=20, n_test=10):
    loaders = []
    for i, classes in enumerate(tasks):
        split = []
        for n, seed in ((n_train, 10 + i), (n_test, 100 + i)):
            X, y = clustered(classes, n, seed=seed, spread=2.5)
            split.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=7))
        loaders.append({'train_eval': split[0], 'val': split[1]})
    return loaders


def test_sweep_reproduces_online_heads(tmp_path):
    engine = _import_engine()
    torch.manual_seed(0)
    tasks = [[0, 1, 2], [3, 4, 5]]
    loaders = _loaders(tasks)
    model, original = FakeModel(), FakeOriginal()
    eargs = Namespace(print_freq=1000, task_inc=False, output_dir='', num_tasks=2, density_dump_dir=str(tmp_path))

    # online: consolidate + evaluate after each task, as train_and_evaluate does
    heads = DensityHeads(make_args(), D)
    metrics = HeadMetrics(2)
    for t in range(2):
        engine.consolidate_density_heads(model, original, loaders[t]['train_eval'], 'cpu', t, heads)
        for i in range(t + 1):
            st = engine.evaluate(model, original, loaders[i]['val'], 'cpu', task_id=i, args=eargs, density=heads)
            for h, a in st['density'].items():
                metrics.update(h, i, t, a)
        online = metrics.end_task(t)
        engine.dump_density_features(model, original, loaders, 'cpu', t, None, eargs)
    assert sorted(os.listdir(tmp_path)) == ['task1.pt', 'task2.pt']

    out = tmp_path / 'sweep.csv'
    args = sw.get_args(['--dump_dir', str(tmp_path), '--out', str(out), '--device', 'cpu', '--rank', '8',
                        '--ranks', '2', '4', '8', '--alphas', '0.5', '1', '--eps', '1e-4', '1e-3',
                        '--opt_ranks', '4', '--opt_iters', '0', '3', '--shrinks', '0.1', '0.5',
                        '--log2_weights', '-2', '2', '--dual_step', '1', '--checkpoints', 'all'])
    sw.run_sweep(args)
    rows = sw.load_rows(str(out))
    online_cfg = {
        'linear': dict(family='linear'),
        'linear_seen': dict(family='linear_seen'),
        'prompted_ncm_whitened': dict(family='ncm_whitened', source='prompted'),
        'frozen_lda': dict(family='lda', source='frozen', shrink=0.1),
        'frozen_lda_fusion': dict(family='lda_fusion', source='frozen', shrink=0.1, w=1.0),
        'prompted_pgm': dict(family='pgm', source='prompted', center='none', rank=8, alpha=1.0, eps=1e-4),
        'frozen_pgm_r2': dict(family='pgm', source='frozen', center='none', rank=2, alpha=1.0, eps=1e-4),
        'prompted_pgm_fusion_r4': dict(family='pgm_fusion', source='prompted', center='none', rank=4, alpha=1.0,
                                       eps=1e-4, w=1.0),
        'dual_pgm_fusion': dict(family='dual_pgm_fusion', center='none', rank=8, alpha=1.0, eps=1e-4, w=1.0, w2=1.0),
    }
    for name, cfg in online_cfg.items():
        match = [r for r in rows if sw._match(r, **cfg)]
        assert len(match) == 1, (name, len(match))
        r = match[0]
        for m, om in (('acc', 'acc'), ('inc_acc', 'incremental_acc'), ('forget', 'forgetting')):
            assert abs(r[m] - online[name][om]) < 1e-9, (name, m, r[m], online[name][om])
    fams = {r['family'] for r in rows}
    assert {'opt', 'dual_opt_fusion', 'dual_lda_fusion', 'dual_ncm_whitened_fusion'} <= fams
    assert any(r['center'] == 'task1' for r in rows)
    assert os.path.exists(tmp_path / 'sweep_summary.txt')

    # compare mode runs on two CSVs with the same grid
    sw.compare([str(out), str(out)], top=3)
