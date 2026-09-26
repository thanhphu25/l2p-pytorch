"""Quantum-state-discrimination (QSD) class heads for rehearsal-free CIL.

Each seen class c is stored as a trace-one, rank-r density matrix
sigma_c = U_c diag(lambda_c) U_c^T built from unit-normalized features of its
training images. A test feature x is the pure state |x><x|. The read-outs
share this memory, so one training run yields a controlled ablation:

  ncm          : cosine to the normalized class mean (classical prototype baseline)
  ncm_centered : ncm after subtracting the mean of the seen class means
  ncm_white    : ncm after whitening by the PGM's S^-1/2 (whitening without Born)
  fidelity     : Born rule without a measurement design, p_c ~ <x|sigma_c|x>
  pgm          : pretty-good measurement over all seen classes,
                 A_c = (sigma_c + eps I) / M, S = sum_c A_c,
                 E_c = S^-1/2 A_c S^-1/2, p_c = <x|E_c|x>  (sum_c E_c = I)
  pgm_r<k>     : the same PGM with every state truncated to its top k eigenpairs
  lda          : classical control, Gaussian posterior with a shared covariance,
                 Sigma = within-class scatter / n + ridge * tr(Sigma) * I

The PGM ridge is eps on trace-one states, so both measurements are regularized
by the same fraction of their trace (lda_ridge defaults to eps).

Class states are written once, when their task ends, and never updated; the
measurements are rebuilt from all stored states as classes arrive. Stored data
are D x r eigenvectors, r eigenvalues and one mean per class plus one shared
D x D scatter matrix for LDA, not images. Everything is simulated with
real-valued PyTorch; no quantum hardware.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


READOUTS = ('ncm', 'ncm_centered', 'ncm_white', 'fidelity', 'pgm', 'lda')


class DensityClassBank(nn.Module):
    def __init__(self, num_classes, dim, rank=32, eps=1e-4, pgm_ranks=(), lda_ridge=None):
        super().__init__()
        if not 1 <= rank <= dim or eps <= 0:
            raise ValueError('Require 1 <= rank <= dim and eps > 0')
        pgm_ranks = tuple(sorted({int(value) for value in pgm_ranks}))
        if any(not 1 <= value < rank for value in pgm_ranks):
            raise ValueError('density_pgm_ranks must lie in [1, density_rank)')
        self.rank, self.eps, self.pgm_ranks = rank, eps, pgm_ranks
        self.lda_ridge = eps if lda_ridge is None else lda_ridge
        if self.lda_ridge <= 0:
            raise ValueError('Require lda_ridge > 0')
        self.register_buffer('vectors', torch.zeros(num_classes, dim, rank))
        self.register_buffer('values', torch.zeros(num_classes, rank))
        self.register_buffer('means', torch.zeros(num_classes, dim))
        # Mean of the unit-normalized features, NOT renormalized (LDA / centering).
        self.register_buffer('raw_means', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))
        # Shared within-class scatter, summed over classes in float64.
        self.register_buffer('scatter', torch.zeros(dim, dim, dtype=torch.float64))
        self.register_buffer('scatter_count', torch.zeros((), dtype=torch.long))
        self._cache = {}
        self.register_load_state_dict_post_hook(self._invalidate)

    def _invalidate(self, *unused):
        self._cache = {}

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
        mean = x.mean(0)
        self.raw_means[label] = mean.to(self.raw_means.dtype)
        self.means[label] = F.normalize(mean, dim=0).to(self.means.dtype)
        centered = x - mean
        self.scatter += (centered.T @ centered).to(self.scatter.dtype)
        self.scatter_count += len(x)
        self.valid[label] = True
        self._invalidate()

    def _states(self, classes, rank=None):
        """Eigenpairs of the seen states, optionally truncated and renormalized."""
        vectors, values = self.vectors[classes], self.values[classes]
        if rank is not None:
            vectors, values = vectors[..., :rank], values[:, :rank]
            values = values / values.sum(-1, keepdim=True).clamp_min(1e-12)
        return vectors, values

    def _cached(self, name, classes, build):
        key = (name, tuple(classes.tolist()), self.vectors.device, self.vectors.dtype)
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def _pgm_root(self, classes, rank=None):
        """S^-1/2 for the uniform-prior PGM, cached until the class set changes."""
        def build():
            # W W^T / M = mean_c sigma_c; ridge eps is shared by every outcome.
            vectors, values = self._states(classes, rank)
            weights = vectors.double() * values.double().sqrt().unsqueeze(1)
            weights = weights.permute(1, 0, 2).flatten(1)
            average = weights @ weights.T / len(classes)
            eigenvalues, eigenvectors = torch.linalg.eigh(average)
            eigenvalues = eigenvalues.clamp_min(0) + self.eps
            root = (eigenvectors * eigenvalues.rsqrt()) @ eigenvectors.T
            return root.to(self.vectors.dtype)
        return self._cached(('pgm', rank), classes, build)

    def _lda(self, classes):
        """Weights and biases of the shared-covariance Gaussian discriminant."""
        def build():
            covariance = self.scatter / self.scatter_count.clamp_min(1)
            # The floor only matters for degenerate one-sample classes (zero scatter).
            ridge = self.lda_ridge * covariance.diagonal().sum().clamp_min(1e-6)
            identity = torch.eye(len(covariance), device=covariance.device, dtype=covariance.dtype)
            means = self.raw_means[classes].double()
            weight = torch.linalg.solve(covariance + ridge * identity, means.T)
            bias = -0.5 * (means * weight.T).sum(-1)
            return weight.to(self.vectors.dtype), bias.to(self.vectors.dtype)
        return self._cached('lda', classes, build)

    def _spread(self, seen, classes):
        full = seen.new_full((len(seen), self.valid.numel()), float('-inf'))
        full[:, classes] = seen
        return full

    @staticmethod
    def _energy(states, vectors, values):
        # <v|sigma_c|v> = sum_k lambda_ck (u_ck . v)^2, for every seen class.
        return (torch.einsum('bd,cdk->bck', states, vectors).square() * values).sum(-1)

    def _pgm_log_probs(self, x, classes, rank=None):
        vectors, values = self._states(classes, rank)
        y = x @ self._pgm_root(classes, rank)  # S^-1/2 is symmetric
        born = (self._energy(y, vectors, values) + self.eps * y.square().sum(-1, keepdim=True)) / len(classes)
        # Completeness makes sum_c p_c = |x|^2 = 1; renormalize only roundoff.
        born = born.clamp_min(1e-12)
        return (born / born.sum(-1, keepdim=True)).log()

    def scores(self, features, readouts=READOUTS):
        """Per-readout scores over ALL classes; unseen classes get -inf.

        fidelity, pgm, pgm_r<k> and lda are log-probabilities over seen classes,
        so they can be fused with log-softmax logits; ncm* are cosine similarities.
        """
        if not self.valid.any():
            raise ValueError('No class density states stored yet')
        classes = self.valid.nonzero(as_tuple=True)[0]
        x = F.normalize(features.to(self.vectors.dtype), dim=-1)
        seen = {}
        if 'ncm' in readouts:
            seen['ncm'] = x @ self.means[classes].T
        if 'ncm_centered' in readouts:
            means = self.raw_means[classes]
            center = means.mean(0)
            seen['ncm_centered'] = F.normalize(x - center, dim=-1) @ F.normalize(means - center, dim=-1).T
        if 'ncm_white' in readouts:
            root = self._pgm_root(classes)
            seen['ncm_white'] = (F.normalize(x @ root, dim=-1)
                                 @ F.normalize(self.raw_means[classes] @ root, dim=-1).T)
        if 'fidelity' in readouts:
            vectors, values = self._states(classes)
            fidelity = self._energy(x, vectors, values).clamp_min(1e-12)
            seen['fidelity'] = (fidelity / fidelity.sum(-1, keepdim=True)).log()
        if 'pgm' in readouts:
            seen['pgm'] = self._pgm_log_probs(x, classes)
            for rank in self.pgm_ranks:
                seen['pgm_r{}'.format(rank)] = self._pgm_log_probs(x, classes, rank)
        if 'lda' in readouts:
            weight, bias = self._lda(classes)
            seen['lda'] = (x @ weight + bias).log_softmax(-1)
        return {name: self._spread(value, classes) for name, value in seen.items()}


class DensityHeads(nn.Module):
    """One class bank per feature source (frozen ViT CLS and/or prompted)."""

    def __init__(self, sources, num_classes, dim, rank=32, eps=1e-4, fusion_weight=1.0,
                 pgm_ranks=(), lda_ridge=None):
        super().__init__()
        if not sources or not set(sources) <= {'frozen', 'prompted'}:
            raise ValueError('density_sources must be a subset of {frozen, prompted}')
        self.fusion_weight = fusion_weight
        self.banks = nn.ModuleDict({
            source: DensityClassBank(num_classes, dim, rank, eps, pgm_ranks, lda_ridge)
            for source in sources})

    def head_logits(self, linear_logits, features):
        """Return {head_name: logits}; `features` maps source -> [B, D]."""
        seen = next(iter(self.banks.values())).valid
        linear = linear_logits.masked_fill(~seen, float('-inf')).log_softmax(-1)
        heads = {}
        for source, bank in self.banks.items():
            scores = bank.scores(features[source])
            for readout, value in scores.items():
                heads['{}_{}'.format(source, readout)] = value
            # Product of experts: L2P classifier x Born probabilities of the PGM,
            # with the classical LDA posterior as the matched control.
            heads['{}_fusion'.format(source)] = linear + self.fusion_weight * scores['pgm']
            heads['{}_lda_fusion'.format(source)] = linear + self.fusion_weight * scores['lda']
            for rank in bank.pgm_ranks:
                heads['{}_fusion_r{}'.format(source, rank)] = (
                    linear + self.fusion_weight * scores['pgm_r{}'.format(rank)])
        return heads


def add_density_head_args(parser):
    parser.add_argument('--density_head', action='store_true',
                        help='also evaluate QSD density-matrix class heads (training is unchanged)')
    parser.add_argument('--density_sources', nargs='+', default=['frozen', 'prompted'],
                        choices=['frozen', 'prompted'])
    parser.add_argument('--density_rank', type=int, default=32)
    parser.add_argument('--density_eps', type=float, default=1e-4)
    parser.add_argument('--density_fusion_weight', type=float, default=1.0)
    parser.add_argument('--density_pgm_ranks', type=int, nargs='*', default=[8, 16],
                        help='also evaluate the PGM with states truncated to these ranks')
    parser.add_argument('--density_lda_ridge', type=float, default=None,
                        help='LDA ridge as a fraction of the covariance trace (default: density_eps)')
