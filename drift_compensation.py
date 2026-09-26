"""Heisenberg-picture compensation of feature drift for old-class classifiers.

L2P's prompt pool is shared across tasks, so features of old classes drift
while their classifier rows are frozen by the training class mask. In the
Schroedinger picture the state evolves, f -> f U; in the Heisenberg picture
the observable evolves instead so that expectation values are unchanged.
Here the drift of the current task's TRAIN features across its own training
(before -> after) is fitted and the SAME transformation is applied to every
old-class classifier row, keeping old logits w . f + b consistent.

  none    : L2P classifier (baseline, same run)
  shift   : translation only, f_old ~ f_new - delta     (SDC-like control)
  affine  : ridge least squares, f_old ~ f_new A + c    (free linear control)
  unitary : rigid motion, f_old ~ (f_new - mu_new) R + mu_old, R orthogonal
            (orthogonal Procrustes; the Heisenberg-picture map)

Training is unchanged: compensated rows live in buffers and are only used
by extra evaluation heads. New classes enter every head with the trained
row; afterwards only compensation changes them. No images are stored.
"""
import torch
from torch import nn


MODES = ('shift', 'affine', 'unitary')


def fit_drift(before, after, ridge=1e-3):
    """Fit maps from AFTER-task features back to BEFORE-task features.

    Returns {mode: (rotation [D, D] or None, offset [D])} such that
    before ~ after @ rotation + offset (rotation None means identity).
    """
    before, after = before.double(), after.double()
    mean_before, mean_after = before.mean(0), after.mean(0)
    centred_before, centred_after = before - mean_before, after - mean_after
    maps = {'shift': (None, mean_before - mean_after)}

    # Ridge toward the identity: before - after ~ after_c @ delta + c.
    gram = centred_after.T @ centred_after
    scale = gram.diagonal().mean().clamp_min(1e-12)
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    delta = torch.linalg.solve(gram + ridge * scale * identity,
                               centred_after.T @ (centred_before - centred_after))
    rotation = identity + delta
    maps['affine'] = (rotation, mean_before - mean_after @ rotation)

    # Orthogonal Procrustes: argmin_R |centred_after R - centred_before|, R^T R = I.
    left, _, right = torch.linalg.svd(centred_after.T @ centred_before)
    rotation = left @ right
    maps['unitary'] = (rotation, mean_before - mean_after @ rotation)
    return maps


def drift_diagnostics(before, after, maps):
    """Relative drift and the residual left by each map (lower is better)."""
    before, after = before.double(), after.double()
    norm = before.norm().clamp_min(1e-12)
    out = {'drift': float((after - before).norm() / norm)}
    for mode, (rotation, offset) in maps.items():
        mapped = after if rotation is None else after @ rotation
        out['residual_' + mode] = float((mapped + offset - before).norm() / norm)
    return out


class HeisenbergHeads(nn.Module):
    def __init__(self, num_classes, dim, ridge=1e-3):
        super().__init__()
        self.ridge = ridge
        self.register_buffer('weight', torch.zeros(len(MODES), num_classes, dim))
        self.register_buffer('bias', torch.zeros(len(MODES), num_classes))
        self.register_buffer('seen', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def compensate(self, before, after):
        """Evolve every stored (old-class) row with the fitted drift map."""
        # Features are collected on CPU; fit and diagnose on the buffers' device.
        before, after = before.to(self.weight.device), after.to(self.weight.device)
        maps = fit_drift(before, after, self.ridge)
        old = self.seen.nonzero(as_tuple=True)[0]
        if len(old):
            for index, mode in enumerate(MODES):
                rotation, offset = maps[mode]
                weight = self.weight[index, old].double()
                # w . f_old + b = (R w) . f_new + (b + w . offset)
                new_weight = weight if rotation is None else weight @ rotation.T
                self.bias[index, old] += (weight @ offset).to(self.bias.dtype)
                self.weight[index, old] = new_weight.to(self.weight.dtype)
        return drift_diagnostics(before, after, maps)

    @torch.no_grad()
    def add_classes(self, classes, head):
        """Copy the freshly trained classifier rows of this task's classes."""
        classes = torch.as_tensor(list(classes), device=self.weight.device)
        if self.seen[classes].any():
            raise ValueError('Classes were already added to the Heisenberg heads')
        self.weight[:, classes] = head.weight[classes].detach().to(self.weight)
        self.bias[:, classes] = head.bias[classes].detach().to(self.bias)
        self.seen[classes] = True

    def head_logits(self, logits, features):
        """{mode: logits}; seen classes use compensated rows, others the live head."""
        out = {}
        seen = self.seen.nonzero(as_tuple=True)[0]
        for index, mode in enumerate(MODES):
            value = logits.clone()
            value[:, seen] = features @ self.weight[index, seen].T + self.bias[index, seen]
            out[mode] = value
        return out


def add_drift_args(parser):
    parser.add_argument('--drift_heads', action='store_true',
                        help='also evaluate Heisenberg drift-compensated classifiers (training unchanged)')
    parser.add_argument('--drift_ridge', type=float, default=1e-3)
