"""Pool results_summary.json files and check the stage-1 decision gates.

    python compare_heads.py /kaggle/working/out_cm_*/results_summary.json

Runs are grouped by prompt selection (batchwise vote or per image). Every
difference is paired: both heads come from the same run, hence the same model.

  G1  prompted_fusion - linear            >= +0.5 on every batchwise seed
  G2  mean(<src>_fusion - <src>_lda_fusion) >= +0.3 (PGM beats matched LDA)
  G3  prompted_fusion - linear            >  0   on every per-image seed
"""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (('final_avg_acc', 'Final'), ('avg_incremental_acc', 'Inc'), ('forgetting', 'Forget'))


def load(paths):
    runs = []
    for path in paths:
        summary = json.loads(Path(path).read_text())
        if summary.get('status') not in ('completed', 'evaluated') or 'density_heads' not in summary:
            print('skip (incomplete or no density heads): {}'.format(path))
            continue
        runs.append(summary)
    return runs


def paired(runs, head, reference):
    return np.array([run['density_heads'][head]['final_avg_acc']
                     - run['density_heads'][reference]['final_avg_acc'] for run in runs])


def fmt(values):
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return '{:6.2f} +- {:4.2f}'.format(values.mean(), std)


def report(runs, label):
    seeds = [run['seed'] for run in runs]
    print('\n=== {} | {} run(s), seeds {} ==='.format(label, len(runs), seeds))
    heads = list(runs[0]['density_heads'])
    print('{:<26}'.format('head') + ''.join('{:>17}'.format(name) for _, name in METRICS)
          + '{:>17}'.format('d vs linear'))
    for head in heads:
        row = '{:<26}'.format(head)
        for key, _ in METRICS:
            row += '{:>17}'.format(fmt([run['density_heads'][head][key] for run in runs]))
        row += '{:>17}'.format(fmt(paired(runs, head, 'linear')))
        print(row)


def gates(groups):
    print('\n=== Decision gates ===')
    batchwise, per_image = groups.get(True, []), groups.get(False, [])
    if batchwise:
        delta = paired(batchwise, 'prompted_fusion', 'linear')
        print('G1 prompted_fusion - linear per seed: {} -> {}'.format(
            np.round(delta, 2).tolist(), 'PASS' if (delta >= 0.5).all() else 'FAIL'))
        for source in ('prompted', 'frozen'):
            delta = paired(batchwise, source + '_fusion', source + '_lda_fusion')
            print('G2 {}_fusion - {}_lda_fusion: per seed {}, mean {:.2f} -> {}'.format(
                source, source, np.round(delta, 2).tolist(), delta.mean(),
                'PASS' if delta.mean() >= 0.3 else 'FAIL'))
    if per_image:
        delta = paired(per_image, 'prompted_fusion', 'linear')
        print('G3 per-image prompted_fusion - linear per seed: {} -> {}'.format(
            np.round(delta, 2).tolist(), 'PASS' if (delta > 0).all() else 'FAIL'))
    if len(batchwise) < 3:
        print('note: G1/G2 were planned over 3 batchwise seeds; have {}'.format(len(batchwise)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('summaries', nargs='+')
    runs = load(parser.parse_args().summaries)
    if not runs:
        raise SystemExit('No completed runs with density heads')
    groups = {}
    for run in runs:
        groups.setdefault(bool(run.get('batchwise_prompt', True)), []).append(run)
    for key in sorted(groups, reverse=True):
        report(groups[key], 'batchwise prompt vote' if key else 'per-image prompts')
    gates(groups)


if __name__ == '__main__':
    main()
