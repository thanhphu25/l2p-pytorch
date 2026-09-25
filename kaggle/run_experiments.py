"""Run the L2P prompt-gating comparison, one run per free GPU.

Every (variant, seed) pair is a separate `main.py` process with its own output dir. Runs whose summary.json already
covers all tasks are skipped, so the script can be re-launched after an interruption.

    python kaggle/run_experiments.py --data-path /tmp/data --out /kaggle/working/results --amp
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_complete(run_dir):
    try:
        with open(os.path.join(run_dir, 'summary.json')) as f:
            summary = json.load(f)
        return summary['completed_tasks'] == summary['num_tasks']
    except (OSError, ValueError, KeyError):
        return False


def last_line(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [l for l in f.read().decode(errors='replace').splitlines() if l.strip()]
        return lines[-1][:220] if lines else ''
    except OSError:
        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='cifar100_l2p', choices=['cifar100_l2p', 'five_datasets_l2p'])
    parser.add_argument('--variants', nargs='+', default=['none', 'softmax', 'quantum', 'linear'],
                        choices=['none', 'softmax', 'linear', 'quantum'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--gpus', type=int, default=None, help='number of GPUs to use (default: all visible)')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--poll', type=int, default=300, help='seconds between progress reports')
    args, extra = parser.parse_known_args()  # anything else is passed through to main.py

    n_gpus = args.gpus
    if n_gpus is None:
        import torch
        n_gpus = torch.cuda.device_count()
    assert n_gpus > 0, 'no GPU visible'
    slots = [gpu for gpu in range(n_gpus) for _ in range(args.jobs_per_gpu)]
    num_workers = max(2, (os.cpu_count() or 4) // len(slots))

    jobs = queue.Queue()
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = os.path.join(args.out, f'{variant}_s{seed}')
            if is_complete(run_dir):
                print(f'skip {run_dir} (complete)')
                continue
            cmd = [sys.executable, 'main.py', args.config, '--model', 'vit_base_patch16_224',
                   '--data-path', args.data_path, '--output_dir', run_dir, '--seed', str(seed),
                   '--prompt_gating', variant, '--save_ckpt', 'none', '--num_workers', str(num_workers),
                   '--print_freq', '100'] + (['--amp'] if args.amp else []) + extra
            jobs.put((run_dir, cmd))

    running, failed = {}, []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                run_dir, cmd = jobs.get_nowait()
            except queue.Empty:
                return
            os.makedirs(run_dir, exist_ok=True)
            log_path = os.path.join(run_dir, 'train.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
            start = time.time()
            print(f'[gpu{gpu}] start {run_dir}\n    ' + ' '.join(cmd), flush=True)
            with lock:
                running[run_dir] = (gpu, log_path, start)
            with open(log_path, 'a') as log:
                code = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                running.pop(run_dir)
                if code != 0:
                    failed.append(run_dir)
            status = 'done' if code == 0 else f'FAILED (exit {code}), see {log_path}'
            print(f'[gpu{gpu}] {status} {run_dir} in {(time.time() - start) / 60:.1f} min', flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in slots]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=args.poll / len(threads))
        with lock:
            for run_dir, (gpu, log_path, start) in running.items():
                print(f'[gpu{gpu}] {os.path.basename(run_dir)} {(time.time() - start) / 60:.0f} min | {last_line(log_path)}',
                      flush=True)
    if failed:
        print('failed runs:', *failed, sep='\n  ')
        sys.exit(1)


if __name__ == '__main__':
    main()
