import json
import unittest
from types import SimpleNamespace

import numpy as np

from result_summary import build_results_summary


class ResultSummaryTest(unittest.TestCase):
    def test_build_qsd_summary(self):
        args = SimpleNamespace(
            prompt_router='qsd', dataset='Split-CIFAR100', seed=10961,
            num_tasks=2, qsd_state_dim=16, qsd_rank=4, qsd_cls_mix=0.5,
            qsd_cosine_tau=0.1, qsd_memory_size=10,
            qsd_retention_coeff=0.1, qsd_no_cosine_prior=False)
        task_summaries = [
            {
                'task': 1, 'avg_acc1': 90.0, 'avg_acc5': 99.0,
                'avg_loss': 0.3, 'forgetting': 0.0,
                'backward_transfer': 0.0, 'train_router_metrics': {},
            },
            {
                'task': 2, 'avg_acc1': 85.0, 'avg_acc5': 98.0,
                'avg_loss': 0.5, 'forgetting': 6.0,
                'backward_transfer': -6.0,
                'train_router_metrics': {'QSDStr': 0.25},
            },
        ]
        accuracy_matrix = np.array([[90.0, 84.0], [0.0, 86.0]])

        summary = build_results_summary(
            args, task_summaries, accuracy_matrix, status='completed')

        self.assertEqual(summary['method'], 'QSD-Prompt')
        self.assertEqual(summary['final_avg_acc'], 85.0)
        self.assertEqual(summary['avg_incremental_acc'], 87.5)
        self.assertEqual(summary['final_per_task_acc'], [84.0, 86.0])
        self.assertEqual(summary['final_router_metrics']['QSDStr'], 0.25)

        # The payload must remain directly serializable for the Kaggle cell.
        loaded = json.loads(json.dumps(summary))
        self.assertEqual(loaded['forgetting'], 6.0)


if __name__ == '__main__':
    unittest.main()
