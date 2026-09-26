"""Quantum-state-discrimination (QSD) class heads for rehearsal-free CIL.

Each seen class c is stored as a trace-one, rank-r density matrix
sigma_c = U_c diag(lambda_c) U_c^T built from unit-normalized features of its
training images. A test feature x is the pure state |x><x|. Three read-outs
share this memory, so one training run yields a controlled ablation:

  ncm      : cosine to the normalized class mean (classical prototype baseline)
  fidelity : Born rule without a measurement design, p_c ~ <x|sigma_c|x>
  pgm      : pretty-good measurement over all seen classes,
             A_c = (sigma_c + eps I) / M, S = sum_c A_c,
             E_c = S^-1/2 A_c S^-1/2, p_c = <x|E_c|x>  (sum_c E_c = I)

Class states are written once, when their task ends, and never updated; the
measurement is rebuilt from all stored states as classes arrive. Stored data
are D x r eigenvectors, r eigenvalues and one mean per class, not images.
Everything is simulated with real-valued PyTorch; no quantum hardware.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


READOUTS = ('ncm', 'fidelity', 'pgm')


class DensityClassBank(nn.Module):
    def __init__(self, num_classes, dim, rank=32, eps=1e-4):
        super().__init__()
        if not 1 <= rank <= dim or eps <= 0:
            raise ValueError('Require 1 <= rank <= dim and eps > 0')
        self.rank, self.eps = rank, eps
        self.register_buffer('vectors', torch.zeros(num_classes, dim, rank))
        self.register_buffer('values', torch.zeros(num_classes, rank))
        self.register_buffer('means', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))
        self._root, self._root_key = None, None
        self.register_load_state_dict_post_hook(self._invalidate)

    def _invalidate(self, *unused):
        self._root, self._root_key = None, None

    @torch.no_grad()
    def add_class(self, label, features):
        """Store the class density matrix; previously stored classes are immutable."""
        label = int(label)
        if self.valid[label]:
            raise ValueError('Class {} already has a density state'.format(label))
        if len(features) == 0:
            raise ValueError('Cannot build a density state from zero samples')
        x = F.normalize(features.to(self.vectors.device, torch.float64), dim=-1)
        # sigma = x^T x / n; its eigenvectors are the right singular vectors.
        _, singular, right = torch.linalg.svd(x / math.sqrt(len(x)), full_matrices=False)
        rank = min(self.rank, len(singular))
        values = singular[:rank].square()
        self.vectors[label].zero_()
        self.values[label].zero_()
        self.vectors[label, :, :rank] = right[:rank].T.to(self.vectors.dtype)
        self.values[label, :rank] = (values / values.sum()).to(self.values.dtype)
        self.means[label] = F.normalize(x.mean(0), dim=0).to(self.means.dtype)
        self.valid[label] = True
        self._invalidate()

    def _pgm_root(self):
        """S^-1/2 for the uniform-prior PGM, cached until the class set changes."""
        classes = self.valid.nonzero(as_tuple=True)[0]
        key = (tuple(classes.tolist()), self.vectors.device, self.vectors.dtype)
        if self._root_key != key:
            # W W^T / M = mean_c sigma_c; ridge eps is shared by every outcome.
            weights = self.vectors[classes].double() * self.values[classes].double().sqrt().unsqueeze(1)
            weights = weights.permute(1, 0, 2).flatten(1)
            average = weights @ weights.T / len(classes)
            eigenvalues, eigenvectors = torch.linalg.eigh(average)
            eigenvalues = eigenvalues.clamp_min(0) + self.eps
            root = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
            self._root, self._root_key = root.to(self.vectors.dtype), key
        return self._root

    def scores(self, features, readouts=READOUTS):
        """Per-readout scores over ALL classes; unseen classes get -inf.

        fidelity and pgm are log-probabilities over seen classes, so they can
        be fused with log-softmax logits; ncm is a cosine similarity.
        """
        if not self.valid.any():
            raise ValueError('No class density states stored yet')
        classes = self.valid.nonzero(as_tuple=True)[0]
        x = F.normalize(features.to(self.vectors.dtype), dim=-1)
        vectors, values = self.vectors[classes], self.values[classes]
        out = {}

        def spread(seen):
            full = seen.new_full((len(x), self.valid.numel()), float('-inf'))
            full[:, classes] = seen
            return full

        def energy(states):
            # <v|sigma_c|v> = sum_k lambda_ck (u_ck . v)^2, for every seen class.
            return (torch.einsum('bd,cdk->bck', states, vectors).square() * values).sum(-1)

        if 'ncm' in readouts:
            out['ncm'] = spread(x @ self.means[classes].T)
        if 'fidelity' in readouts:
            fidelity = energy(x).clamp_min(1e-12)
            out['fidelity'] = spread((fidelity / fidelity.sum(-1, keepdim=True)).log())
        if 'pgm' in readouts:
            y = x @ self._pgm_root()  # S^-1/2 is symmetric
            born = (energy(y) + self.eps * y.square().sum(-1, keepdim=True)) / len(classes)
            # Completeness makes sum_c p_c = |x|^2 = 1; renormalize only roundoff.
            born = born.clamp_min(1e-12)
            out['pgm'] = spread((born / born.sum(-1, keepdim=True)).log())
        return out


class DensityHeads(nn.Module):
    """One class bank per feature source (frozen ViT CLS and/or prompted)."""

    def __init__(self, sources, num_classes, dim, rank=32, eps=1e-4, fusion_weight=1.0):
        super().__init__()
        if not sources or not set(sources) <= {'frozen', 'prompted'}:
            raise ValueError('density_sources must be a subset of {frozen, prompted}')
        self.fusion_weight = fusion_weight
        self.banks = nn.ModuleDict({
            source: DensityClassBank(num_classes, dim, rank, eps) for source in sources})

    def head_logits(self, linear_logits, features):
        """Return {head_name: logits}; `features` maps source -> [B, D]."""
        seen = next(iter(self.banks.values())).valid
        linear = linear_logits.masked_fill(~seen, float('-inf')).log_softmax(-1)
        heads = {}
        for source, bank in self.banks.items():
            scores = bank.scores(features[source])
            for readout, value in scores.items():
                heads['{}_{}'.format(source, readout)] = value
            # Product of experts: L2P classifier x Born probabilities of the PGM.
            heads['{}_fusion'.format(source)] = linear + self.fusion_weight * scores['pgm']
        return heads


def add_density_head_args(parser):
    parser.add_argument('--density_head', action='store_true',
                        help='also evaluate QSD density-matrix class heads (training is unchanged)')
    parser.add_argument('--density_sources', nargs='+', default=['frozen', 'prompted'],
                        choices=['frozen', 'prompted'])
    parser.add_argument('--density_rank', type=int, default=32)
    parser.add_argument('--density_eps', type=float, default=1e-4)
    parser.add_argument('--density_fusion_weight', type=float, default=1.0)
