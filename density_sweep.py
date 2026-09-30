"""
Offline sweep of density heads on features dumped by

    python main.py <config> --eval --output_dir OUT --density_dump_dir DUMP

Class states are rebuilt from the dumped train splits with ClassBank, exactly as consolidate_density_heads
does online, so the original heads (prompted_pgm_r8, frozen_lda, ...) reproduce the online table.

Families (every fusion is log_softmax(linear over seen) + w * log p_head, dual_* uses w, w2 for frozen, prompted):
  classical  linear, ncm_whitened, lda (shrinkage grid); their fusions, the LDA weight tuned on its own grid
  pgm        B: power family rho^alpha (alpha = 1 is the PGM), ridge eps, readout rank
             C: states of centered features, x -> normalize(normalize(x) - g), g = mean of task-1 train features
  opt        A: minimum-error measurement, fixed-point iteration of Jezek, Rehacek, Fiurasek (PRA 65, 060301)
             Pi_c <- G^-1 q rho_c Pi_c rho_c q G^-1,  G^2 = sum_c q^2 rho_c Pi_c rho_c,  started from the PGM;
             (ridge --opt_ridge on G^2); iters = 0 is the PGM readout without its class-independent eps floor

Usage:
    python density_sweep.py --dump_dir DUMP_CIFAR --out cifar.csv
    python density_sweep.py --dump_dir DUMP_CUB --out cub.csv
    python density_sweep.py --compare cifar.csv cub.csv
"""
import argparse
import csv
import glob
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from density_head import ClassBank

SOURCES = ('frozen', 'prompted')
FIELDS = ['family', 'source', 'center', 'rank', 'alpha', 'eps', 'iters', 'shrink', 'w', 'w2']
METRICS = ['acc', 'inc_acc', 'forget']
CLASSICAL = {'linear', 'linear_seen', 'ncm_whitened', 'lda', 'ncm_whitened_fusion', 'lda_fusion', 'dual_ncm_whitened_fusion',
             'dual_lda_fusion'}
STANDALONE = {'linear', 'linear_seen', 'ncm_whitened', 'lda', 'pgm', 'opt'}
_TINY = 1e-300


def get_args(argv=None):
    p = argparse.ArgumentParser('density head sweep')
    p.add_argument('--dump_dir', default='')
    p.add_argument('--out', default='density_sweep.csv')
    p.add_argument('--compare', nargs=2, metavar='CSV', help='compare two sweep CSVs (e.g. CIFAR and CUB)')
    p.add_argument('--checkpoints', default='last', choices=['last', 'all'],
                   help='last: accuracy after the final task only; all: also IncAcc and Forgetting (slower)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--rank', default=32, type=int, help='stored rank per class')
    p.add_argument('--ranks', default=[1, 2, 4, 8, 16, 32], type=int, nargs='+')
    p.add_argument('--alphas', default=[0.25, 0.5, 1.0, 2.0, 4.0], type=float, nargs='+')
    p.add_argument('--eps', default=[1e-5, 1e-4, 1e-3, 1e-2], type=float, nargs='+')
    p.add_argument('--centers', default=['none', 'task1'], nargs='+', choices=['none', 'task1'])
    p.add_argument('--opt_ranks', default=[2, 8, 32], type=int, nargs='*')
    p.add_argument('--opt_iters', default=[0, 1, 2, 5, 10, 20, 50, 100], type=int, nargs='+')
    p.add_argument('--opt_eps', default=1e-4, type=float, help='eps of the PGM the iteration starts from')
    p.add_argument('--opt_ridge', default=1e-8, type=float,
                   help='ridge added to G^2, relative to its trace; above ~1e-6 the iteration stops being an ascent')
    p.add_argument('--shrinks', default=[0.01, 0.03, 0.1, 0.3, 0.6, 0.9], type=float, nargs='+')
    p.add_argument('--log2_weights', default=[-10, 8], type=int, nargs=2, help='fusion weights 2^k, k in [lo, hi]')
    p.add_argument('--dual_step', default=2, type=int, help='dual fusions use every n-th fusion weight')
    p.add_argument('--chunk', default=2048, type=int)
    p.add_argument('--top', default=12, type=int)
    return p.parse_args(argv)


def _key(**kw):
    return tuple(kw.get(f, '') for f in FIELDS)


def key_name(key):
    d = dict(zip(FIELDS, key))
    parts = [d['source'] + '_' + d['family'] if d['source'] else d['family']]
    for f, tag in [('center', 'c'), ('rank', 'r'), ('alpha', 'a'), ('eps', 'eps'), ('iters', 'it'),
                   ('shrink', 'sh'), ('w', 'w'), ('w2', 'w2')]:
        v = d[f]
        if v == '' or (f == 'center' and v == 'none'):
            continue
        parts.append(f'{tag}={float(v):g}' if f != 'center' else f'c={v}')
    return ' '.join(parts)


def _inv_sqrt(S):
    e, V = torch.linalg.eigh((S + S.T) / 2)
    return (V * e.clamp_min(_TINY).rsqrt()) @ V.T


class Whitening:
    """S = (1/M) sum_c rho_c^alpha (eigenvalues renormalized) of the first `rank` eigenpairs, and its eigh."""

    def __init__(self, bank, rank, alpha, device):
        U = bank.U[:, :, :rank].to(device, torch.float64)
        lam = bank.lam[:, :rank].to(device, torch.float64)
        lam = lam / lam.sum(1, keepdim=True).clamp_min(_TINY)
        if alpha != 1:
            lam = torch.where(lam > 0, lam.clamp_min(_TINY) ** alpha, torch.zeros_like(lam))
            lam = lam / lam.sum(1, keepdim=True).clamp_min(_TINY)
        self.M, self.D, self.r = U.shape
        self.U, self.lam = U, lam
        self.U_flat = U.permute(1, 0, 2).reshape(self.D, -1)
        W = self.U_flat * lam.reshape(1, -1).sqrt()
        self.e, self.V = torch.linalg.eigh(W @ W.T / self.M)
        self.e = self.e.clamp_min(0)

    def S_mh(self, eps):
        return (self.V * (self.e + eps).rsqrt()) @ self.V.T


def pgm_logp(wh, xn, eps_list, chunk):
    """log p_c of the complete PGM for every eps; alpha = 1 matches ClassBank.pgm_probs."""
    out = {}
    for eps in eps_list:
        S_mh = wh.S_mh(eps)
        parts = []
        for x in xn.split(chunk):
            y = x @ S_mh
            proj = (y @ wh.U_flat).view(x.shape[0], wh.M, wh.r)
            p = ((proj ** 2 * wh.lam).sum(-1) + eps * (y ** 2).sum(-1, keepdim=True)) / wh.M
            parts.append(torch.log((p / p.sum(1, keepdim=True)).clamp_min(_TINY)))
        out[eps] = torch.cat(parts)
    return out


def opt_measurements(bank, rank, eps, ridge, iters, device):
    """Minimum-error measurement for the class states rho_c (uniform priors), by the fixed-point iteration
    of Jezek et al. started from the PGM with ridge eps. Every element stays Pi_c = Q U_c K_c U_c^T Q (Q symmetric, K_c r x r),
    so memory is O(M D r). Returns {iteration: (Q, K)} and {iteration: sum_c q Tr(rho_c Pi_c)}."""
    wh = Whitening(bank, rank, 1.0, device)
    U, lam, U_flat, M, D, r = wh.U, wh.lam, wh.U_flat, wh.M, wh.D, wh.r
    eye = torch.eye(D, device=device, dtype=torch.float64)
    Q = wh.S_mh(eps)
    K = torch.diag_embed(lam / M)
    snaps, success = {}, {}
    for k in range(max(iters) + 1):
        QU = (Q @ U_flat).view(D, M, r).permute(1, 0, 2)
        B = U.transpose(1, 2) @ QU                      # U_c^T Q U_c
        A = B @ K @ B                                   # U_c^T Pi_c U_c
        success[k] = (lam * torch.diagonal(A, dim1=1, dim2=2)).sum().item() / M
        if k in iters:
            snaps[k] = (Q, K)
        if k == max(iters):
            break
        N = lam[:, :, None] * A * lam[:, None, :] / M ** 2  # q^2 rho_c Pi_c rho_c in the basis U_c
        G2 = (U @ N).permute(1, 0, 2).reshape(D, -1) @ U_flat.T
        G2 = (G2 + G2.T) / 2
        Q = _inv_sqrt(G2 + ridge * torch.trace(G2) * eye)
        K = N
    return wh, snaps, success


def opt_logp(wh, Q, K, xn, chunk):
    parts = []
    for x in xn.split(chunk):
        Z = ((x @ Q) @ wh.U_flat).view(x.shape[0], wh.M, wh.r)
        s = (torch.einsum('nmr,mrs->nms', Z, K) * Z).sum(-1).clamp_min(0)
        parts.append(torch.log((s / s.sum(1, keepdim=True).clamp_min(_TINY)).clamp_min(_TINY)))
    return torch.cat(parts)


def lda_scores(bank, f, shrinks, device):
    cov = bank.scatter.to(device) / max(bank.n_scatter, 1)
    iso = torch.trace(cov) / bank.dim
    e, V = torch.linalg.eigh(cov)
    mu = bank.mu_raw.to(device, torch.float64)
    out = {}
    for sh in shrinks:
        P = (V / ((1 - sh) * e + sh * iso)) @ V.T
        Pm = mu @ P
        out[sh] = f @ Pm.T - 0.5 * (Pm * mu).sum(1)
    return out


class Recorder:
    """Accuracy matrices acc[key][test_task, checkpoint], same conventions as HeadMetrics."""

    def __init__(self, num_tasks):
        self.T = num_tasks
        self.acc = {}
        self.cols = set()

    def begin(self, t, test, cls, device):
        self.t = t
        self.cols.add(t)
        lookup = torch.full((int(max(cls)) + 1,), -1, dtype=torch.long)
        lookup[torch.tensor(cls)] = torch.arange(len(cls))
        target = torch.cat([d['target'] for d in test])
        self.tgt = lookup[target].to(device)
        assert (self.tgt >= 0).all(), 'test target outside the seen classes'
        task = torch.cat([torch.full((len(d['target']),), i) for i, d in enumerate(test)])
        self.onehot = F.one_hot(task, t + 1).to(device, torch.float64)
        self.onehot /= self.onehot.sum(0, keepdim=True)
        logits = torch.cat([d['logits'] for d in test]).to(device, torch.float64)
        self.log_lin = F.log_softmax(logits[:, torch.tensor(cls, device=device)], dim=1)
        self.linear_correct = (logits.argmax(1).cpu() == target).double().to(device)  # over all columns, = Acc@1

    def _store(self, keys, pred=None, correct=None):
        correct = (pred == self.tgt).double() if correct is None else correct
        acc = 100.0 * correct @ self.onehot   # (heads, tasks)
        for key, a in zip(keys, acc.cpu().numpy()):
            m = self.acc.setdefault(key, np.zeros((self.T, self.T)))
            m[:self.t + 1, self.t] = a

    def standalone(self, key, scores):
        self._store([key], scores.argmax(1)[None])

    def fusion(self, base, L, weights, extra=None):
        """log_lin + w * L for every w; `extra` is an already weighted second term (dual fusions)."""
        base_scores = self.log_lin if extra is None else self.log_lin + extra
        for ws in [weights[i:i + 4] for i in range(0, len(weights), 4)]:
            w = torch.tensor(ws, device=L.device, dtype=torch.float64)[:, None, None]
            pred = (base_scores[None] + w * L[None]).argmax(-1)
            self._store([base(w_) for w_ in ws], pred)

    def dual(self, base, L1, L2, weights):
        for w1 in weights:
            self.fusion(lambda w2: base(w1, w2), L2, weights, extra=w1 * L1)

    def rows(self):
        last = max(self.cols)
        for key, m in self.acc.items():
            row = dict(zip(FIELDS, key))
            row['acc'] = float(np.mean(m[:last + 1, last]))
            if len(self.cols) == self.T:
                row['inc_acc'] = float(np.mean([np.mean(m[:t + 1, t]) for t in range(self.T)]))
                row['forget'] = float(np.mean((np.max(m, axis=1) - m[:, last])[:last])) if last > 0 else 0.0
            else:
                row['inc_acc'] = row['forget'] = ''
            yield row


def evaluate_checkpoint(rec, t, banks, centers_g, test, args, log):
    dev = args.device
    rec.begin(t, test, banks[(args.sources[0], 'none')].classes, dev)
    weights = [2.0 ** k for k in range(args.log2_weights[0], args.log2_weights[1] + 1)]
    dual_w = weights[::args.dual_step]
    ranks = sorted({k for k in args.ranks if 0 < k <= args.rank})
    dual = set(args.sources) == set(SOURCES)

    def heads(family, cfg, lps):
        """standalone + fusion per source, dual fusion over the two sources; lps: source -> log p"""
        for s, lp in lps.items():
            rec.standalone(_key(family=family, source=s, **cfg), lp)
            rec.fusion(lambda w: _key(family=f'{family}_fusion', source=s, w=w, **cfg), lp, weights)
        if dual:
            rec.dual(lambda w1, w2: _key(family=f'dual_{family}_fusion', w=w1, w2=w2, **cfg),
                     lps['frozen'], lps['prompted'], dual_w)

    rec._store([_key(family='linear')], correct=rec.linear_correct[None])
    rec.standalone(_key(family='linear_seen'), rec.log_lin)
    feats, xc = {}, {}
    for s in args.sources:
        feats[s] = torch.cat([d[s] for d in test]).to(dev, torch.float64)
        xn = F.normalize(feats[s], dim=1)
        for c in args.centers:
            xc[(s, c)] = xn if c == 'none' else F.normalize(xn - centers_g[s].to(dev), dim=1)

    # classical controls; fusing a score s with weight w equals fusing log_softmax(w s), so w is the temperature
    cos = {}
    for s in args.sources:
        bank = banks[(s, 'none')]
        mu = bank.mu_norm.to(dev, torch.float64)
        S_mh = bank.measurement(bank.rank, dev)['S_mh']
        cos[s] = F.normalize(xc[(s, 'none')] @ S_mh, dim=1) @ F.normalize(mu @ S_mh, dim=1).T
    heads('ncm_whitened', {}, cos)
    lda = {s: lda_scores(banks[(s, 'none')], feats[s], args.shrinks, dev) for s in args.sources}
    for sh in args.shrinks:
        heads('lda', dict(shrink=sh), {s: lda[s][sh] for s in args.sources})
    del cos, lda

    for c in args.centers:
        # B (and C): PGM family
        for r in ranks:
            for a in (args.alphas if r > 1 else [1.0]):
                lps = {s: pgm_logp(Whitening(banks[(s, c)], r, a, dev), xc[(s, c)], args.eps, args.chunk)
                       for s in args.sources}
                for eps in args.eps:
                    heads('pgm', dict(center=c, rank=r, alpha=a, eps=eps), {s: lps[s][eps] for s in args.sources})
        # A: minimum-error measurement
        for r in [k for k in args.opt_ranks if 0 < k <= args.rank]:
            lps = {}
            for s in args.sources:
                t0 = time.time()
                wh, snaps, success = opt_measurements(banks[(s, c)], r, args.opt_eps, args.opt_ridge, args.opt_iters, dev)
                log(f'  opt {s} c={c} r={r}: train success ' +
                    ' '.join(f'it{k}={success[k]:.4f}' for k in sorted(snaps)) + f'  ({time.time() - t0:.1f}s)')
                lps[s] = {k: opt_logp(wh, Q, K, xc[(s, c)], args.chunk) for k, (Q, K) in snaps.items()}
            for k in sorted(lps[args.sources[0]]):
                heads('opt', dict(center=c, rank=r, eps=args.opt_eps, iters=k), {s: lps[s][k] for s in args.sources})


def run_sweep(args):
    files = glob.glob(os.path.join(args.dump_dir, 'task*.pt'))
    files = sorted(files, key=lambda p: int(re.findall(r'task(\d+)\.pt$', p)[0]))
    assert files, f'no task*.pt in {args.dump_dir}'
    first = torch.load(files[0])
    T = first['num_tasks']
    assert len(files) == T, f'{len(files)} dumps for {T} tasks'
    args.sources = [s for s in SOURCES if s in first['train']]
    D = first['train'][args.sources[0]].shape[1]
    banks = {(s, c): ClassBank(D, args.rank) for s in args.sources for c in args.centers}
    centers_g = {s: F.normalize(first['train'][s].double(), dim=1).mean(0) for s in args.sources}
    del first
    rec = Recorder(T)
    eval_at = set(range(T)) if args.checkpoints == 'all' else {T - 1}
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    for t, path in enumerate(files):
        d = torch.load(path)
        tr = d['train']
        for c in torch.unique(tr['labels']).tolist():
            idx = tr['labels'] == c
            for s in args.sources:
                x = tr[s][idx].double()
                banks[(s, 'none')].add_class(c, x)
                if 'task1' in args.centers:
                    banks[(s, 'task1')].add_class(c, F.normalize(x, dim=1) - centers_g[s])
        if t not in eval_at:
            continue
        t0 = time.time()
        log(f'checkpoint {t + 1}/{T}: {len(banks[(args.sources[0], "none")])} classes')
        evaluate_checkpoint(rec, t, banks, centers_g, d['test'], args, log)
        log(f'  {len(rec.acc)} heads, {time.time() - t0:.1f}s')

    rows = list(rec.rows())
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS + METRICS)
        wr.writeheader()
        wr.writerows(rows)
    log(f'wrote {len(rows)} heads to {args.out}')
    summarize(rows, log)
    with open(os.path.splitext(args.out)[0] + '_summary.txt', 'w') as fh:
        fh.write('\n'.join(log_lines) + '\n')


def load_rows(path):
    with open(path) as fh:
        rows = [_norm(r) for r in csv.DictReader(fh)]
    for r in rows:
        for m in METRICS:
            r[m] = float(r[m]) if r[m] != '' else None
    return rows


def _norm(row):
    """Stringify config fields the way load_rows does, so rows from a run and from a CSV compare equal."""
    out = dict(row)
    for f in ('rank', 'iters'):
        if out.get(f, '') != '':
            out[f] = str(int(float(out[f])))
    for f in ('alpha', 'eps', 'shrink', 'w', 'w2'):
        if out.get(f, '') != '':
            out[f] = repr(float(out[f]))
    return out


def _match(row, **kw):
    return all(row[k] == v for k, v in _norm(kw).items() if k in FIELDS) if kw else True


def _best(rows, pred):
    cand = [r for r in rows if pred(r)]
    return max(cand, key=lambda r: r['acc']) if cand else None


def _fmt(row, ref=None):
    if row is None:
        return '-'
    s = f'{row["acc"]:7.2f}'
    if ref is not None:
        s += f'  {row["acc"] - ref:+6.2f}'
    if row.get('forget') not in (None, ''):
        s += f'  inc {row["inc_acc"]:.2f} fgt {row["forget"]:.2f}'
    return s


# online head name -> config, for checking that the offline rebuild reproduces the online table
ONLINE = {
    'linear': dict(family='linear'),
    'linear_seen': dict(family='linear_seen'),
    'frozen_ncm_whitened': dict(family='ncm_whitened', source='frozen'),
    'prompted_ncm_whitened': dict(family='ncm_whitened', source='prompted'),
    'frozen_lda': dict(family='lda', source='frozen', shrink=0.1),
    'prompted_lda': dict(family='lda', source='prompted', shrink=0.1),
    'frozen_lda_fusion': dict(family='lda_fusion', source='frozen', shrink=0.1, w=1.0),
    'frozen_pgm': dict(family='pgm', source='frozen', center='none', rank=32, alpha=1.0, eps=1e-4),
    'prompted_pgm': dict(family='pgm', source='prompted', center='none', rank=32, alpha=1.0, eps=1e-4),
    'prompted_pgm_r8': dict(family='pgm', source='prompted', center='none', rank=8, alpha=1.0, eps=1e-4),
    'frozen_pgm_fusion_r8': dict(family='pgm_fusion', source='frozen', center='none', rank=8, alpha=1.0,
                                 eps=1e-4, w=1.0),
    'dual_pgm_fusion': dict(family='dual_pgm_fusion', center='none', rank=32, alpha=1.0, eps=1e-4, w=1.0, w2=1.0),
}


def summarize(rows, log):
    rows = [_norm(r) for r in rows]
    log('\n[Online heads, rebuilt offline: must match the online table]')
    for name, cfg in ONLINE.items():
        r = _best(rows, lambda x: _match(x, **cfg))
        log(f'  {name:<28}{_fmt(r)}')

    lda_ref = _best(rows, lambda r: r['family'] == 'lda')['acc']
    ldaf = _best(rows, lambda r: r['family'] in ('lda_fusion', 'dual_lda_fusion'))
    ldaf_ref = ldaf['acc'] if ldaf else lda_ref
    groups = [
        ('Standalone', lda_ref, [
            ('linear', lambda r: r['family'] == 'linear'),
            ('ncm_whitened (best source)', lambda r: r['family'] == 'ncm_whitened'),
            ('lda, shrink 0.1 (online)', lambda r: r['family'] == 'lda' and r['shrink'] == repr(0.1)),
            ('lda, shrink tuned  <- REF', lambda r: r['family'] == 'lda'),
            ('pgm original (r, a=1, eps=1e-4)', lambda r: r['family'] == 'pgm' and r['center'] == 'none'
             and r['alpha'] == repr(1.0) and r['eps'] == repr(1e-4)),
            ('B: pgm, alpha tuned', lambda r: r['family'] == 'pgm' and r['center'] == 'none'
             and r['eps'] == repr(1e-4)),
            ('B: pgm, eps tuned', lambda r: r['family'] == 'pgm' and r['center'] == 'none'
             and r['alpha'] == repr(1.0)),
            ('B: pgm, all tuned', lambda r: r['family'] == 'pgm' and r['center'] == 'none'),
            ('C: pgm centered, all tuned', lambda r: r['family'] == 'pgm' and r['center'] == 'task1'),
            ('A: opt, iters>0', lambda r: r['family'] == 'opt' and r['iters'] != '0'),
        ]),
        ('Fusion with linear', ldaf_ref, [
            ('lda_fusion, shrink .1 w=1 (online)', lambda r: r['family'] == 'lda_fusion'
             and r['shrink'] == repr(0.1) and r['w'] == repr(1.0)),
            ('lda_fusion, tuned', lambda r: r['family'] == 'lda_fusion'),
            ('lda fusions incl. dual  <- REF', lambda r: r['family'] in ('lda_fusion', 'dual_lda_fusion')),
            ('ncm_whitened fusions incl. dual', lambda r: r['family'] in ('ncm_whitened_fusion',
                                                                          'dual_ncm_whitened_fusion')),
            ('pgm fusion original, w tuned', lambda r: r['family'] in ('pgm_fusion', 'dual_pgm_fusion')
             and r['center'] == 'none' and r['alpha'] == repr(1.0) and r['eps'] == repr(1e-4)),
            ('B: pgm fusions, all tuned', lambda r: r['family'] in ('pgm_fusion', 'dual_pgm_fusion')
             and r['center'] == 'none'),
            ('C: centered pgm fusions', lambda r: r['family'] in ('pgm_fusion', 'dual_pgm_fusion')
             and r['center'] == 'task1'),
            ('A: opt fusions, iters>0', lambda r: r['family'] in ('opt_fusion', 'dual_opt_fusion')
             and r['iters'] != '0'),
        ]),
    ]
    for title, ref, items in groups:
        log(f'\n[{title}]   acc  vs LDA ref ({ref:.2f})   best config')
        for label, pred in items:
            r = _best(rows, pred)
            log(f'  {label:<38}{_fmt(r, ref):<40}{key_name(tuple(r[f] for f in FIELDS)) if r else ""}')


def compare(paths, top):
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    data = [load_rows(p) for p in paths]
    by_key = [{tuple(r[f] for f in FIELDS): r for r in rows} for rows in data]
    keys = set(by_key[0]) & set(by_key[1])
    print(f'{len(keys)} head configs present in both {names[0]} and {names[1]}')
    verdict = []
    for title, lda_fams, standalone in [('Standalone', {'lda'}, True),
                                        ('Fusion with linear', {'lda_fusion', 'dual_lda_fusion'}, False)]:
        in_cat = lambda fam: (fam in STANDALONE) == standalone
        lda = [max((r for r in rows if r['family'] in lda_fams), key=lambda r: r['acc']) for rows in data]
        classical = [max((r for r in rows if r['family'] in CLASSICAL and in_cat(r['family'])),
                         key=lambda r: r['acc']) for rows in data]
        print(f'\n=== {title} ===')
        for n, l, c in zip(names, lda, classical):
            print(f'  {n}: LDA tuned per dataset {l["acc"]:.2f} ({key_name(tuple(l[f] for f in FIELDS))}); '
                  f'best classical {c["acc"]:.2f} ({key_name(tuple(c[f] for f in FIELDS))})')
        cands = []
        for k in keys:
            fam = k[0]
            if fam in CLASSICAL or not in_cat(fam):
                continue
            m = [by_key[i][k]['acc'] - lda[i]['acc'] for i in range(2)]
            cands.append((min(m), m, k))
        cands.sort(reverse=True)
        print(f'  Same config on both datasets, ranked by the worse margin over the tuned LDA:')
        print(f'  {"":6}{"min":>7}{names[0][:10]:>11}{names[1][:10]:>11}   config')
        for mn, m, k in cands[:top]:
            flag = 'WIN' if mn > 0 else ''
            print(f'  {flag:<6}{mn:+7.2f}{m[0]:+11.2f}{m[1]:+11.2f}   {key_name(k)}')
        print('  Each dataset tuned on its own (upper bound, same privilege LDA gets):')
        for fam in sorted({k[0] for _, _, k in cands}):
            best = [max((r for r in rows if r['family'] == fam), key=lambda r: r['acc']) for rows in data]
            print(f'    {fam:<18}' + '  '.join(f'{n}: {b["acc"]:.2f} ({b["acc"] - l["acc"]:+.2f})'
                                               for n, b, l in zip(names, best, lda)))
        verdict.append((title, cands[0] if cands else None))
    print('\n=== Verdict: some quantum variant beats the tuned LDA on both datasets with one config? ===')
    for title, c in verdict:
        ok = c is not None and c[0] > 0
        print(f'  {title}: {"YES" if ok else "NO"}' + (f'  best: {key_name(c[2])} (min margin {c[0]:+.2f})'
                                                      if c else ''))


if __name__ == '__main__':
    args = get_args()
    if args.compare:
        compare(args.compare, args.top)
    else:
        run_sweep(args)
