# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import utils
from compositional_prompt import CompositionalPrompt
from result_summary import (
    ROUTER_METRICS,
    build_results_summary,
    extract_router_metrics,
    save_results_summary,
)


def _prompt_query_from_original(output):
    """Return frozen ViT patch tokens for QSD without retaining its graph."""
    tokens = output.get('x')
    if tokens is not None and tokens.ndim == 3 and tokens.shape[1] > 1:
        return tokens[:, 1:]
    return tokens


def _update_router_metrics(metric_logger, output, batch_size):
    names = {
        'qsd_entropy': 'QSDEnt',
        'qsd_purity': 'QSDPur',
        'qsd_strength': 'QSDStr',
        'route_entropy': 'RouteEnt',
        'qsd_retention_loss': 'QSDRet',
        'comp_retention_loss': 'CompKL',
        'comp_prompt_loss': 'CompMSE',
        'old_new_mass': 'OldNewMass',
        'route_peak': 'RoutePeak',
        'active_components': 'Components',
        'memory_count': 'MemCount',
        'povm_error': 'POVMError',
    }
    for output_name, meter_name in names.items():
        if output_name in output:
            metric_logger.meters[meter_name].update(
                output[output_name].detach().item(), n=batch_size)

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                original_output = original_model(input)
                cls_features = original_output['pre_logits']
                prompt_query_tokens = _prompt_query_from_original(original_output)
            else:
                cls_features = None
                prompt_query_tokens = None
        
        output = model(
            input, task_id=task_id, cls_features=cls_features,
            prompt_query_tokens=prompt_query_tokens, train=set_training_mode)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']
        if 'qsd_retention_loss' in output:
            loss = loss + args.qsd_retention_coeff * output['qsd_retention_loss']
        if 'comp_retention_loss' in output:
            loss = loss + args.comp_retention_coeff * output['comp_retention_loss']
            loss = loss + args.comp_prompt_coeff * output['comp_prompt_loss']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        _update_router_metrics(metric_logger, output, input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                original_output = original_model(input)
                cls_features = original_output['pre_logits']
                prompt_query_tokens = _prompt_query_from_original(original_output)
            else:
                cls_features = None
                prompt_query_tokens = None
            
            output = model(
                input, task_id=task_id, cls_features=cls_features,
                prompt_query_tokens=prompt_query_tokens)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            _update_router_metrics(metric_logger, output, input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    router_metrics = {name: [] for name in ROUTER_METRICS}

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        for name in router_metrics:
            if name in test_stats:
                router_metrics[name].append(float(test_stats[name]))

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    forgetting = 0.0
    backward = 0.0
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    task_summary = {
        'task': int(task_id + 1),
        'avg_acc1': float(avg_stat[0]),
        'avg_acc5': float(avg_stat[1]),
        'avg_loss': float(avg_stat[2]),
        'forgetting': float(forgetting),
        'backward_transfer': float(backward),
        'current_task_acc1': float(acc_matrix[task_id, task_id]),
        'per_task_acc1': [float(acc_matrix[i, task_id]) for i in range(task_id + 1)],
        'eval_router_metrics': {
            name: float(np.mean(values))
            for name, values in router_metrics.items() if values
        },
    }
    return test_stats, task_summary

@torch.no_grad()
def consolidate_compositional_memory(prompt, original_model, loader, device, labels, budget):
    """Use only this task's TRAIN loader; keep a bounded candidate set per class."""
    original_model.eval()
    candidates = {int(label): [[], [], 0] for label in labels}
    print('Collecting train-only density prototypes (up to {} candidates/class)'.format(budget))
    for inputs, targets in loader:
        output = original_model(inputs.to(device, non_blocking=True))
        queries, states = prompt.encode_states(
            output['pre_logits'], _prompt_query_from_original(output))
        for label in targets.unique().tolist():
            if label not in candidates:
                raise ValueError('Prototype loader contains a class outside the current task')
            query_list, state_list, count = candidates[label]
            remaining = budget - count
            if remaining <= 0:
                continue
            indices = (targets == label).nonzero(as_tuple=True)[0][:remaining].to(device)
            query_list.append(queries[indices].cpu())
            state_list.append(states[indices].cpu())
            candidates[label][2] += len(indices)
        if all(value[2] >= budget for value in candidates.values()):
            break
    for label, (queries, states, count) in candidates.items():
        if not count:
            raise ValueError('No training prototype candidates found for class {}'.format(label))
        prompt.consolidate_class(label, torch.cat(queries), torch.cat(states))
    print('Stored density prototypes:', int(prompt.memory_valid.sum()))


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    task_summaries = []
    compositional = isinstance(getattr(model_without_ddp, 'prompt', None), CompositionalPrompt)

    for task_id in range(args.num_tasks):
        if compositional:
            model_without_ddp.prompt.begin_task(task_id)
            # Rebuild AFTER freezing old banks, also rebinding the scheduler.
            optimizer = create_optimizer(args, model_without_ddp)
            lr_scheduler = create_scheduler(args, optimizer)[0] if args.sched != 'constant' else None
       # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(cur_start, cur_end))
                    prev_idx = (slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if not compositional and task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        if compositional:
            consolidate_compositional_memory(
                model_without_ddp.prompt, original_model, data_loader[task_id]['train'],
                device, class_mask[task_id], args.comp_candidates_per_class)

        test_stats, task_summary = evaluate_till_now(
            model=model, original_model=original_model, data_loader=data_loader,
            device=device, task_id=task_id, class_mask=class_mask,
            acc_matrix=acc_matrix, args=args)
        task_summary['train_router_metrics'] = extract_router_metrics(train_stats)
        task_summaries.append(task_summary)

        # Retain only the task's mean density state and measurement outcome.
        # Consolidate before checkpointing so evaluation-only runs restore the
        # exact continual-learning state.
        if hasattr(model_without_ddp, 'prompt') and not compositional:
            model_without_ddp.prompt.consolidate_router()

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

            # Overwrite one compact file after every task. It remains useful if
            # a long Kaggle run is interrupted before all tasks finish.
            running_summary = build_results_summary(
                args, task_summaries, acc_matrix, status='running')
            save_results_summary(running_summary, args.output_dir)

    return build_results_summary(
        args, task_summaries, acc_matrix, status='completed')
