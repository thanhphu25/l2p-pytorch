"""Collect summary.json files written by engine.train_and_evaluate into one comparison table.

    python kaggle/summarize.py --root /kaggle/working/results
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ORDER = ['none', 'softmax', 'linear', 'quantum']
NAMES = {'none': 'L2P', 'softmax': 'L2P + softmax gate', 'linear': 'L2P + linear gate (param-matched)',
         'quantum': 'L2P + quantum gate'}
METRICS = [('final_avg_acc', 'Final avg acc'), ('avg_incremental_acc', 'Avg incremental acc'),
           ('forgetting', 'Forgetting'), ('epoch_time_mean', 'Epoch time (s)')]


def fmt(values, digits=2):
    if len(values) == 1:
        return f'{values[0]:.{digits}f}'
    return f'{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default=None, help='where to write results.md / results.csv (default: --root)')
    args = parser.parse_args()
    out = args.out or args.root

    runs = []
    for path in sorted(glob.glob(os.path.join(args.root, '*', 'summary.json'))):
        with open(path) as f:
            s = json.load(f)
        s['run'] = os.path.basename(os.path.dirname(path))
        s['test_gate_dev'] = s['history'][-1].get('test_GateDev', float('nan'))
        s['test_gate_ent'] = s['history'][-1].get('test_GateEnt', float('nan'))
        runs.append(s)
    if not runs:
        print('no summary.json under', args.root)
        return

    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        keys = ['run', 'prompt_gating', 'seed', 'amp', 'completed_tasks', 'num_tasks', 'final_avg_acc',
                'avg_incremental_acc', 'forgetting', 'epoch_time_mean', 'train_time_total', 'n_trainable_params',
                'n_gate_params', 'test_gate_ent', 'test_gate_dev']
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(runs)

    variants = sorted({r['prompt_gating'] for r in runs}, key=ORDER.index)
    groups = {v: [r for r in runs if r['prompt_gating'] == v] for v in variants}
    lines = [f"Dataset: {runs[0]['dataset']}, epochs/task: {runs[0]['epochs']}, batch: {runs[0]['batch_size']}, "
             f"AMP: {runs[0]['amp']}", '',
             '| Method | Seeds | ' + ' | '.join(name for _, name in METRICS) +
             ' | Trainable params | Gate params | Gate dev. from softmax (test) |',
             '|' + '---|' * (len(METRICS) + 5)]
    for v in variants:
        g = groups[v]
        partial = [r['run'] for r in g if r['completed_tasks'] < r['num_tasks']]
        seeds = ','.join(str(r['seed']) for r in g) + (f' (partial: {", ".join(partial)})' if partial else '')
        cells = [fmt([r[k] for r in g], 1 if k == 'epoch_time_mean' else 2) for k, _ in METRICS]
        dev = fmt([r['test_gate_dev'] for r in g], 3) if v != 'none' else '-'
        lines.append(f"| {NAMES[v]} | {seeds} | " + ' | '.join(cells) +
                     f" | {g[0]['n_trainable_params']:,} | {g[0]['n_gate_params']:,} | {dev} |")

    # paired differences on shared seeds, the comparison the experiment is about
    def delta(a, b):
        seeds = sorted({r['seed'] for r in groups.get(a, [])} & {r['seed'] for r in groups.get(b, [])})
        if not seeds:
            return None
        pick = lambda v, s, k: next(r[k] for r in groups[v] if r['seed'] == s)
        return {k: [pick(a, s, k) - pick(b, s, k) for s in seeds] for k, _ in METRICS[:3]}, seeds

    lines += ['', 'Paired differences (same seeds; forgetting: lower is better):', '']
    for a, b in [('softmax', 'none'), ('linear', 'none'), ('quantum', 'none'), ('quantum', 'softmax'), ('quantum', 'linear')]:
        d = delta(a, b)
        if d:
            diffs, seeds = d
            lines.append(f'- {a} − {b} (seeds {seeds}): ' +
                         ', '.join(f'{name} {fmt(diffs[k])}' for k, name in METRICS[:3]))
    if all(len(g) == 1 for g in groups.values()):
        lines += ['', 'Single seed only: read small differences with care; the L2P paper reports a std over 3 runs of '
                      '0.04 (Split CIFAR-100) to 0.93 (5-datasets) points.']

    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(out, 'results.md'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
