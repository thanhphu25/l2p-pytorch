"""Layer-wise unitary evolution of the L2P-selected prompt (prefix-tuning).

The prompt P0 retrieved by L2P is treated as an initial state and evolved
through a depth-L "circuit" of layer unitaries,

    P_l = P_{l-1} U_l,   U_l = exp(A_l),   A_l = B_l C_l^T - C_l B_l^T,

separately for the key and value streams. P_l is prepended to the keys and
values of transformer block l. Because every U_l is orthogonal, the Gram
matrix of ALL pool prompts, i.e. the distinguishability (fidelity) between
prompts learned for different tasks, is identical at every layer, even while
the shared circuit is updated for a new task. The controls isolate that
property:

  unitary     : U_l = exp(A_l)              (orthogonal, 2*r*d params / layer / stream)
  linear      : M_l = I + B_l C_l^T         (same parameters, not norm/Gram preserving)
  shared      : P_l = P0                    (no circuit parameters)
  independent : P_l selected from a separate per-layer pool (DualPrompt-like)

Real-valued PyTorch simulation; no quantum hardware.
"""
import math

import torch
from torch import nn


MODES = ('unitary', 'linear', 'shared', 'independent')


class PromptCircuit(nn.Module):
    def __init__(self, dim, layers, mode='unitary', rank=8, pool_size=None, length=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown circuit mode: {}'.format(mode))
        if not layers or len(set(layers)) != len(layers):
            raise ValueError('circuit_layers must be non-empty and unique')
        self.dim, self.mode = dim, mode
        self.layers = sorted(int(layer) for layer in layers)
        depth = len(self.layers)
        if mode in ('unitary', 'linear'):
            if not 1 <= rank <= dim // 2:
                raise ValueError('Require 1 <= circuit_rank <= dim / 2')
            # B = 0 makes every layer map the identity at initialization, so
            # training starts from "copy the retrieved prompt to every layer".
            self.left = nn.Parameter(torch.zeros(2, depth, dim, rank))
            self.right = nn.Parameter(torch.randn(2, depth, dim, rank) / math.sqrt(dim))
        elif mode == 'independent':
            if pool_size is None or length is None:
                raise ValueError('independent mode needs the prompt pool size and length')
            self.pool = nn.Parameter(torch.empty(depth, 2, pool_size, length, dim).uniform_(-1, 1))

    def low_rank_maps(self):
        """(W, K) with layer map I + W K, shapes [2, depth, d, q] and [2, depth, q, d].

        unitary: A = B C^T - C B^T = W M with W = [B, C], M = [C^T; -B^T], and
        exp(W M) = I + W phi(M W) M, phi(X) = sum_k X^k / (k+1)!, obtained
        exactly from exp([[X, I], [0, 0]]) of size 4r instead of a d x d exp.
        """
        if self.mode == 'linear':
            return self.left, self.right.transpose(-1, -2)
        if self.mode != 'unitary':
            return None
        w = torch.cat((self.left, self.right), dim=-1)
        m = torch.cat((self.right, -self.left), dim=-1).transpose(-1, -2)
        x = m @ w
        q = x.shape[-1]
        eye = torch.eye(q, device=x.device, dtype=x.dtype).expand_as(x)
        block = torch.cat((torch.cat((x, eye), -1), torch.zeros_like(torch.cat((x, x), -1))), -2)
        # torch 1.12 may warn "output ... was resized" for batches whose
        # matrices have different norms; values and gradients are correct.
        phi = torch.matrix_exp(block)[..., :q, q:]
        return w, phi @ m

    def layer_maps(self):
        """Dense [2, depth, d, d] maps (tests/inspection); None for shared/independent."""
        factors = self.low_rank_maps()
        if factors is None:
            return None
        identity = torch.eye(self.dim, device=factors[0].device, dtype=factors[0].dtype)
        return identity + factors[0] @ factors[1]

    def forward(self, prompt, prompt_idx=None, pool=None):
        """prompt: [B, T, d] retrieved by L2P. Returns ({block: [B, 2, T', d]}, diagnostics)."""
        prefixes, diagnostics = {}, {}
        if self.mode == 'independent':
            if prompt_idx is None:
                raise ValueError('independent mode needs the L2P prompt indices')
            for depth, layer in enumerate(self.layers):
                selected = self.pool[depth][:, prompt_idx]  # [2, B, top_k, length, d]
                prefixes[layer] = selected.flatten(2, 3).transpose(0, 1)
            return prefixes, diagnostics
        factors = self.low_rank_maps()
        states = [prompt, prompt]
        for depth, layer in enumerate(self.layers):
            if factors is not None:
                states = [self._evolve(factors, stream, depth, states[stream]) for stream in range(2)]
            prefixes[layer] = torch.stack(states, dim=1)
        if factors is not None:
            with torch.no_grad():
                diagnostics.update(self._geometry(prompt, states, factors, pool))
        return prefixes, diagnostics

    @staticmethod
    def _evolve(factors, stream, depth, states):
        """states @ (I + W K) without forming the d x d map."""
        w, k = factors[0][stream, depth], factors[1][stream, depth]
        return states + (states @ w) @ k

    def _geometry(self, prompt, states, factors, pool):
        """Norm ratio of the deepest prefix and Gram drift of the whole pool."""
        norm = prompt.norm(dim=-1).clamp_min(1e-12)
        ratio = torch.stack([state.norm(dim=-1) / norm for state in states]).mean()
        out = {'circuit_norm_ratio': ratio}
        if pool is not None:
            vectors = pool.reshape(-1, pool.shape[-1])
            gram = vectors @ vectors.T
            drift = []
            for stream in range(2):
                total = vectors
                for depth in range(len(self.layers)):
                    total = self._evolve(factors, stream, depth, total)
                drift.append((total @ total.T - gram).norm() / gram.norm().clamp_min(1e-12))
            out['circuit_gram_drift'] = torch.stack(drift).mean()
        return out


def add_circuit_args(parser):
    parser.add_argument('--circuit_mode', default='none', choices=('none',) + MODES,
                        help='layer-wise prefix prompts derived from the L2P prompt (none = L2P)')
    parser.add_argument('--circuit_layers', default=[1, 2, 3, 4, 5], type=int, nargs='+',
                        help='transformer blocks that receive prefix key/value prompts')
    parser.add_argument('--circuit_rank', default=8, type=int)
