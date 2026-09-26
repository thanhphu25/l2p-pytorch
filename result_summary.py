import json
from pathlib import Path


ROUTER_METRICS = ('QSDEnt', 'QSDPur', 'QSDStr', 'RouteEnt', 'QSDRet',
                  'CompKL', 'CompMSE', 'OldNewMass', 'RoutePeak',
                  'Components', 'MemCount', 'POVMError')


def extract_router_metrics(stats):
    """Select compact QSD diagnostics from a MetricLogger result."""
    return {name: float(stats[name]) for name in ROUTER_METRICS if name in stats}


def build_results_summary(args, task_summaries, acc_matrix, status='running'):
    """Build a JSON-serializable continual-learning experiment summary."""
    completed = len(task_summaries)
    summary = {
        'schema_version': 1,
        'status': status,
        'method': {'qsd': 'QSD-Prompt', 'qsd_comp': 'QSD Compositional Prompt',
                   'cosine_comp': 'Cosine Compositional Prompt'}.get(args.prompt_router, 'L2P'),
        'dataset': args.dataset,
        'seed': int(args.seed),
        'prompt_router': args.prompt_router,
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

    summary['final_new_task_acc'] = float(acc_matrix[completed - 1, completed - 1])
    summary['final_old_task_acc'] = (float(acc_matrix[:completed - 1, completed - 1].mean())
                                     if completed > 1 else None)
    summary['new_minus_old_gap'] = (summary['final_new_task_acc'] - summary['final_old_task_acc']
                                     if completed > 1 else None)

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

    if args.prompt_router in ('qsd_comp', 'cosine_comp'):
        summary['compositional_config'] = {
            name: getattr(args, name) for name in (
                'epochs', 'batch_size', 'length', 'top_k', 'qsd_state_dim', 'qsd_rank',
                'qsd_eps', 'qsd_cls_mix', 'qsd_cosine_tau', 'comp_components_per_task',
                'comp_quantum_mix', 'comp_prototypes_per_class', 'comp_candidates_per_class',
                'comp_memory_batch_size', 'comp_retention_coeff', 'comp_prompt_coeff')}
        summary['compositional_config']['effective_quantum_mix'] = (
            args.comp_quantum_mix if args.prompt_router == 'qsd_comp' else 0.0)
        summary['compositional_config']['active_components'] = completed * args.comp_components_per_task
        summary['compositional_config']['prompt_tokens'] = args.length * args.top_k
        summary['final_router_metrics'] = last.get('eval_router_metrics', {})
        summary['final_train_router_metrics'] = last.get('train_router_metrics', {})
        summary['final_router_metrics_source'] = 'evaluation'

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
    if summary.get('final_old_task_acc') is not None:
        print('Old / new task Acc@1   : {:.4f} / {:.4f}'.format(
            summary['final_old_task_acc'], summary['final_new_task_acc']))
        print('New-minus-old gap      : {:.4f}'.format(summary['new_minus_old_gap']))
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
    if summary_path is not None:
        print('Saved summary          : ' + str(summary_path))
    print(line)
