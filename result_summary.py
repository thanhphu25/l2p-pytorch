import json
from pathlib import Path

import numpy as np


ROUTER_METRICS = ('QSDEnt', 'QSDPur', 'QSDStr', 'RouteEnt', 'QSDRet')


def extract_router_metrics(stats):
    """Select compact QSD diagnostics from a MetricLogger result."""
    return {name: float(stats[name]) for name in ROUTER_METRICS if name in stats}


def matrix_metrics(acc_matrix, completed):
    """Final/incremental accuracy, forgetting and BWT of one CIL accuracy matrix."""
    matrix = np.asarray(acc_matrix, dtype=float)[:completed, :completed]
    last = completed - 1
    forgetting = backward = 0.0
    if completed > 1:
        forgetting = float(np.mean(matrix[:last].max(axis=1) - matrix[:last, last]))
        backward = float(np.mean(matrix[:last, last] - np.diag(matrix)[:last]))
    return {
        'final_avg_acc': float(matrix[:, last].mean()),
        'avg_incremental_acc': float(np.mean([
            matrix[:task + 1, task].mean() for task in range(completed)])),
        'forgetting': forgetting,
        'backward_transfer': backward,
        'final_per_task_acc': [float(value) for value in matrix[:, last]],
    }


def density_head_results(task_summaries, acc_matrix):
    """Rebuild each density head's accuracy matrix from per-task rows."""
    completed = len(task_summaries)
    names = task_summaries[-1].get('head_per_task_acc1', {}) if completed else {}
    if not names:
        return {}
    results = {'linear': matrix_metrics(acc_matrix, completed)}
    for name in names:
        matrix = np.zeros((completed, completed))
        for column, task in enumerate(task_summaries):
            row = task.get('head_per_task_acc1', {}).get(name, [])
            matrix[:len(row), column] = row
        results[name] = matrix_metrics(matrix, completed)
        results[name]['accuracy_matrix'] = matrix.tolist()
    return results


def build_results_summary(args, task_summaries, acc_matrix, status='running'):
    """Build a JSON-serializable continual-learning experiment summary."""
    completed = len(task_summaries)
    summary = {
        'schema_version': 1,
        'status': status,
        'method': 'QSD-Prompt' if args.prompt_router == 'qsd' else 'L2P',
        'dataset': args.dataset,
        'seed': int(args.seed),
        'prompt_router': args.prompt_router,
        'batchwise_prompt': bool(getattr(args, 'batchwise_prompt', True)),
        'tasks_completed': completed,
        'num_tasks': int(args.num_tasks),
        'task_summaries': task_summaries,
    }

    if completed == 0:
        summary['accuracy_matrix'] = []
        return summary

    last = task_summaries[-1]
    summary.update({
        'final_avg_acc': float(last['avg_acc1']),
        'avg_incremental_acc': float(
            sum(task['avg_acc1'] for task in task_summaries) / completed),
        'final_avg_acc5': float(last['avg_acc5']),
        'final_loss': float(last['avg_loss']),
        'forgetting': float(last['forgetting']),
        'backward_transfer': float(last['backward_transfer']),
        'best_avg_acc': float(max(task['avg_acc1'] for task in task_summaries)),
        'final_per_task_acc': [
            float(acc_matrix[i, completed - 1]) for i in range(completed)
        ],
        'accuracy_matrix': [
            [float(acc_matrix[row, column]) for column in range(completed)]
            for row in range(completed)
        ],
    })

    if args.prompt_router == 'qsd':
        summary['qsd_config'] = {
            'state_dim': int(args.qsd_state_dim),
            'rank': int(args.qsd_rank),
            'cls_mix': float(args.qsd_cls_mix),
            'cosine_tau': float(args.qsd_cosine_tau),
            'memory_size': int(args.qsd_memory_size),
            'retention_coeff': float(args.qsd_retention_coeff),
            'no_cosine_prior': bool(args.qsd_no_cosine_prior),
        }
        summary['final_router_metrics'] = last.get('train_router_metrics', {})

    heads = density_head_results(task_summaries, acc_matrix)
    if heads:
        summary['density_config'] = {
            'sources': list(args.density_sources),
            'rank': int(args.density_rank),
            'eps': float(args.density_eps),
            'fusion_weight': float(args.density_fusion_weight),
            'pgm_ranks': [int(rank) for rank in getattr(args, 'density_pgm_ranks', [])],
            'lda_ridge': float(args.density_eps if getattr(args, 'density_lda_ridge', None) is None
                               else args.density_lda_ridge),
        }
        summary['density_heads'] = heads

    return summary


def save_results_summary(summary, output_dir):
    """Atomically write the compact summary and return its path."""
    if not output_dir:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / 'results_summary.json'
    temporary_path = output_path / 'results_summary.json.tmp'
    with temporary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write('\n')
    temporary_path.replace(summary_path)
    return summary_path


def print_results_summary(summary, summary_path=None):
    """Print only the metrics normally needed to compare CIL runs."""
    line = '=' * 66
    print('\n' + line)
    print('FINAL CONTINUAL-LEARNING SUMMARY')
    print(line)
    print('Method / router        : {} / {}'.format(
        summary['method'], summary['prompt_router']))
    print('Dataset / seed         : {} / {}'.format(
        summary['dataset'], summary['seed']))
    print('Tasks completed        : {}/{}'.format(
        summary['tasks_completed'], summary['num_tasks']))
    print('Final average Acc@1    : {:.4f}'.format(summary['final_avg_acc']))
    print('Average incremental Acc: {:.4f}'.format(summary['avg_incremental_acc']))
    print('Final average Acc@5    : {:.4f}'.format(summary['final_avg_acc5']))
    print('Forgetting             : {:.4f}'.format(summary['forgetting']))
    print('Backward transfer      : {:.4f}'.format(summary['backward_transfer']))
    print('Final per-task Acc@1   : {}'.format(
        ', '.join('{:.2f}'.format(value)
                  for value in summary['final_per_task_acc'])))
    if 'total_training_time' in summary:
        print('Total training time    : {}'.format(summary['total_training_time']))
    if summary.get('final_router_metrics'):
        diagnostics = ', '.join(
            '{}={:.4f}'.format(name, value)
            for name, value in summary['final_router_metrics'].items())
        print('QSD diagnostics        : ' + diagnostics)
    if summary.get('density_heads'):
        print('Head (same run)             Final Acc  Inc. Acc  Forgetting')
        for name, result in summary['density_heads'].items():
            print('  {:<25} {:>9.2f} {:>9.2f} {:>11.4f}'.format(
                name, result['final_avg_acc'], result['avg_incremental_acc'],
                result['forgetting']))
    if summary_path is not None:
        print('Saved summary          : ' + str(summary_path))
    print(line)
