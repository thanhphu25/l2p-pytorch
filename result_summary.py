import json
from pathlib import Path


PATCH_METRICS = ('PatchEnt', 'PatchDev', 'PatchAlpha', 'PatchPeak', 'PhaseStd')


def extract_patch_metrics(stats):
    return {name: float(stats[name]) for name in PATCH_METRICS if name in stats}


def build_results_summary(args, task_summaries, acc_matrix, status='running'):
    completed = len(task_summaries)
    method_names = {
        'none': 'L2P',
        'phase': 'L2P + Phase Interference',
        'mlp': 'L2P + Matched MLP',
        'phase_no_encoding': 'L2P + Circuit without Phase Encoding',
    }
    summary = {
        'schema_version': 1,
        'status': status,
        'method': method_names[args.patch_transform],
        'dataset': args.dataset,
        'seed': int(args.seed),
        'patch_transform': args.patch_transform,
        'batchwise_prompt': bool(args.batchwise_prompt),
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
        'final_old_task_acc': last['old_task_acc1'],
        'final_new_task_acc': float(last['current_task_acc1']),
        'final_old_new_gap': last['old_new_gap'],
        'final_per_task_acc': [
            float(acc_matrix[i, completed - 1]) for i in range(completed)
        ],
        'accuracy_matrix': [
            [float(acc_matrix[row, column]) for column in range(completed)]
            for row in range(completed)
        ],
        'final_diagnostics': last.get('train_patch_metrics', {}),
    })
    if args.patch_transform != 'none':
        summary['patch_transform_config'] = {
            'mode': args.patch_transform,
            'latent_dim': int(args.phase_latent_dim),
            'circuit_depth': int(args.phase_circuit_depth),
            'alpha_init': float(args.phase_alpha_init),
            'alpha_max': float(args.phase_alpha_max),
            'circuit_init_std': float(args.phase_circuit_init_std),
        }
    return summary


def save_results_summary(summary, output_dir):
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
    line = '=' * 66
    print('\n' + line)
    print('FINAL CONTINUAL-LEARNING SUMMARY')
    print(line)
    print('Method                 : {}'.format(summary['method']))
    print('Seed / tasks           : {} / {}/{}'.format(
        summary['seed'], summary['tasks_completed'], summary['num_tasks']))
    print('Final average Acc@1    : {:.4f}'.format(summary['final_avg_acc']))
    print('Average incremental Acc: {:.4f}'.format(summary['avg_incremental_acc']))
    print('Forgetting             : {:.4f}'.format(summary['forgetting']))
    print('Backward transfer      : {:.4f}'.format(summary['backward_transfer']))
    if summary.get('final_old_task_acc') is not None:
        print('Old / new task Acc@1   : {:.4f} / {:.4f}'.format(
            summary['final_old_task_acc'], summary['final_new_task_acc']))
        print('New-minus-old gap      : {:.4f}'.format(
            summary['final_old_new_gap']))
    print('Final per-task Acc@1   : {}'.format(
        ', '.join('{:.2f}'.format(value)
                  for value in summary['final_per_task_acc'])))
    if 'total_training_time' in summary:
        print('Total training time    : {}'.format(summary['total_training_time']))
    if summary.get('final_diagnostics'):
        print('Patch diagnostics      : ' + ', '.join(
            '{}={:.4f}'.format(name, value)
            for name, value in summary['final_diagnostics'].items()))
    if summary_path is not None:
        print('Saved summary          : ' + str(summary_path))
    print(line)
