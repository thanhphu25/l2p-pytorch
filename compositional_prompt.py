"""Differentiable quantum-state routing of task-frozen prompt components.

Only prompt tokens are composed. The image encoder and patch tokens are never
adapted. Density matrices and the PGM are simulated with real-valued PyTorch;
this is a quantum-inspired method, not a quantum-hardware implementation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class _InverseSqrt(torch.autograd.Function):
    """SPD inverse square root, with a finite derivative at repeated eigenvalues."""

    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh((matrix + matrix.mT) * 0.5)
        if torch.any(values <= 0):
            raise ValueError('PGM ensemble must be positive definite; increase qsd_eps')
        roots = values.sqrt()
        ctx.save_for_backward(roots, vectors)
        return (vectors * roots.reciprocal().unsqueeze(-2)) @ vectors.mT

    @staticmethod
    def backward(ctx, grad):
        roots, vectors = ctx.saved_tensors
        a, b = roots.unsqueeze(-1), roots.unsqueeze(-2)
        # (a^-1 - b^-1) / (a^2 - b^2), including the a == b limit.
        coefficient = -1.0 / (a * b * (a + b))
        local = vectors.mT @ ((grad + grad.mT) * 0.5) @ vectors
        return vectors @ (coefficient * local) @ vectors.mT


def complete_pgm(states, eps):
    """Uniform-prior PGM with ridge distributed over ALL measurement outcomes.

    A_j=(sigma_j + eps I)/M; S=sum_j A_j; E_j=S^-1/2 A_j S^-1/2.
    Thus sum_j E_j=I even when the original states do not span the space.
    """
    identity = torch.eye(states.shape[-1], device=states.device, dtype=states.dtype)
    operators = (states + eps * identity) / states.shape[-3]
    inverse = _InverseSqrt.apply(operators.sum(dim=-3))
    return inverse.unsqueeze(-3) @ operators @ inverse.unsqueeze(-3)


class CompositionalPrompt(nn.Module):
    def __init__(self, embed_dim, length, heads, num_tasks, components_per_task,
                 num_classes, state_dim=32, rank=4, eps=1e-3, cls_mix=0.5,
                 cosine_tau=0.1, quantum_mix=0.5, prototypes_per_class=4,
                 memory_batch_size=32, router='qsd_comp'):
        super().__init__()
        if router not in ('qsd_comp', 'cosine_comp'):
            raise ValueError('Unknown compositional router')
        if not 2 <= state_dim <= embed_dim or not 1 <= rank <= state_dim:
            raise ValueError('Require 2 <= state_dim <= embed_dim and 1 <= rank <= state_dim')
        if min(length, heads, num_tasks, components_per_task, num_classes,
               prototypes_per_class, memory_batch_size) < 1:
            raise ValueError('Prompt and memory sizes must be positive')
        if eps <= 0 or cosine_tau <= 0 or not 0 <= cls_mix <= 1 or not 0 <= quantum_mix <= 1:
            raise ValueError('Invalid regularization, temperature or mixture coefficient')
        self.length, self.top_k = length, heads
        self.num_tasks, self.components_per_task = num_tasks, components_per_task
        self.pool_size = num_tasks * components_per_task
        self.state_dim, self.eps, self.cls_mix = state_dim, eps, cls_mix
        self.cosine_tau = cosine_tau
        self.quantum_mix = quantum_mix if router == 'qsd_comp' else 0.0
        self.router = router
        self.prototypes_per_class = prototypes_per_class
        self.memory_batch_size = memory_batch_size

        # A buffer cannot drift and is not overwritten by ViT's Linear init.
        projection = torch.empty(embed_dim, state_dim)
        nn.init.orthogonal_(projection)
        self.register_buffer('projection', projection)
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.empty(components_per_task, length, embed_dim).uniform_(-1, 1))
            for _ in range(num_tasks)])
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(heads, components_per_task, state_dim, rank)
                         / math.sqrt(state_dim)) for _ in range(num_tasks)])
        self.register_buffer('active_tasks', torch.tensor(1, dtype=torch.long))
        self.register_buffer('training_task', torch.tensor(0, dtype=torch.long))

        slots = num_classes * prototypes_per_class
        self.register_buffer('memory_states', torch.zeros(slots, state_dim, state_dim))
        self.register_buffer('memory_queries', torch.zeros(slots, state_dim))
        self.register_buffer('memory_probs', torch.zeros(slots, heads, self.pool_size))
        self.register_buffer('memory_valid', torch.zeros(slots, dtype=torch.bool))
        self.register_buffer('memory_task', torch.full((slots,), -1, dtype=torch.long))
        self.register_buffer('memory_counts', torch.zeros(slots, dtype=torch.long))
        self._set_trainable_bank()
        self.register_load_state_dict_post_hook(self._restore_trainable_bank)

    def _set_trainable_bank(self):
        task = int(self.training_task.item())
        for index, (prompts, factors) in enumerate(zip(self.prompts, self.factors)):
            for parameter in (prompts, factors):
                parameter.requires_grad_(index == task)
                parameter.grad = None  # also protect against stale optimizer momentum

    def _restore_trainable_bank(self, module, incompatible_keys):
        self._set_trainable_bank()

    def begin_task(self, task_id):
        if not 0 <= task_id < self.num_tasks:
            raise ValueError('Task index outside configured capacity')
        active = int(self.active_tasks.item())
        if task_id not in (active - 1, active):
            raise ValueError('Tasks must be learned in sequence')
        self.active_tasks.fill_(task_id + 1)
        self.training_task.fill_(task_id)
        self._set_trainable_bank()

    def _unit(self, vectors):
        norm = vectors.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(vectors)
        fallback[..., 0] = 1
        return torch.where(norm > 1e-8, vectors / norm.clamp_min(1e-8), fallback)

    def encode_states(self, cls_features, patch_features):
        if cls_features is None or patch_features is None:
            raise ValueError('Compositional routing requires frozen CLS and patch features')
        query = self._unit(cls_features.detach() @ self.projection)
        patches = self._unit(patch_features.detach() @ self.projection)
        cls_state = query.unsqueeze(-1) * query.unsqueeze(-2)
        patch_state = patches.mT @ patches / patches.shape[-2]
        density = self.cls_mix * cls_state + (1 - self.cls_mix) * patch_state
        return query, density

    def routing_operators(self):
        active = int(self.active_tasks.item())
        factors = torch.cat(list(self.factors[:active]), dim=1)
        # Both controls use all factor parameters, with identical parameter count.
        keys = self._unit(factors.sum(dim=-1))
        measurement = None
        if self.quantum_mix > 0:
            states = factors @ factors.mT
            states = states / states.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None].clamp_min(1e-12)
            measurement = complete_pgm(states, self.eps)
        return keys, measurement

    def route(self, query, density, operators=None):
        keys, measurement = operators if operators is not None else self.routing_operators()
        similarity = torch.einsum('bd,hmd->bhm', query, keys)
        cosine = (similarity / self.cosine_tau).softmax(dim=-1)
        born = None
        if measurement is not None:
            born = torch.einsum('hmij,bji->bhm', measurement, density).clamp_min(0)
            # Only remove floating-point roundoff; completeness is tested BEFORE this.
            born = born / born.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probabilities = (1 - self.quantum_mix) * cosine + self.quantum_mix * born
        else:
            probabilities = cosine
        return probabilities, similarity, born

    def prompt_components(self):
        return torch.cat(list(self.prompts[:int(self.active_tasks.item())]), dim=0)

    @staticmethod
    def compose(probabilities, components):
        return torch.einsum('bhm,mlc->bhlc', probabilities, components)

    def retention_loss(self, operators, components):
        # Current-task prototypes are excluded, including during checkpoint evaluation.
        valid = self.memory_valid & (self.memory_task < self.training_task)
        indices = valid.nonzero(as_tuple=True)[0]
        zero = components.new_zeros(())
        if not indices.numel():
            return zero, zero, zero
        if self.training and indices.numel() > self.memory_batch_size:
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[:self.memory_batch_size]]
        probs, _, _ = self.route(self.memory_queries[indices], self.memory_states[indices], operators)
        targets = self.memory_probs[indices, :, :components.shape[0]]
        # Keep new components in the denominator: conditioning only on old ones
        # would fail to penalize stealing routing mass from previous tasks.
        kl = (targets * (targets.clamp_min(1e-8).log() - probs.clamp_min(1e-8).log())).sum(-1).mean()
        target_prompt = self.compose(targets, components.detach())
        actual_prompt = self.compose(probs, components)
        mse = (actual_prompt - target_prompt).square().mean()
        mse = mse / target_prompt.square().mean().clamp_min(1e-6)
        current_start = int(self.training_task.item()) * self.components_per_task
        new_mass = probs[..., current_start:].sum(-1).mean()
        return kl, mse, new_mass

    def forward(self, x_embed, prompt_mask=None, cls_features=None,
                prompt_query_tokens=None, update_router_memory=False):
        if prompt_mask is not None:
            raise ValueError('Compositional prompts do not accept task-specific routing masks')
        query, density = self.encode_states(cls_features, prompt_query_tokens)
        operators = self.routing_operators()
        probs, similarity, born = self.route(query, density, operators)
        components = self.prompt_components()
        tokens = self.compose(probs, components).flatten(1, 2)
        kl, mse, old_new_mass = self.retention_loss(operators, components)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()
        out = {
            'prompted_embedding': torch.cat((tokens, x_embed), dim=1),
            'total_prompt_len': self.top_k * self.length,
            'route_probabilities': probs,
            'reduce_sim': (probs * similarity).sum(-1).sum(-1).mean(),
            'comp_retention_loss': kl,
            'comp_prompt_loss': mse,
            'route_entropy': entropy,
            'qsd_strength': probs.new_tensor(self.quantum_mix),
            'qsd_purity': density.square().sum((-1, -2)).mean(),
            'old_new_mass': old_new_mass,
            'route_peak': probs.max(dim=-1).values.mean(),
            'active_components': probs.new_tensor(components.shape[0]),
            'memory_count': self.memory_valid.sum().to(probs.dtype),
        }
        if born is not None:
            out['qsd_entropy'] = -(born * born.clamp_min(1e-8).log()).sum(-1).mean()
            identity = torch.eye(self.state_dim, device=probs.device, dtype=probs.dtype)
            out['povm_error'] = (operators[1].sum(1) - identity).abs().max()
        return out

    @torch.no_grad()
    def consolidate_class(self, label, queries, states):
        """Cluster TRAIN states; snapshot targets with the final task router.

        Old targets are immutable. Stored data are density/query centroids and
        probabilities, never input images or full ViT embeddings.
        """
        start = int(label) * self.prototypes_per_class
        if start < 0 or start + self.prototypes_per_class > self.memory_valid.numel():
            raise ValueError('Class label outside configured memory')
        if self.memory_valid[start:start + self.prototypes_per_class].any():
            raise ValueError('Cannot overwrite a previously consolidated class')
        if not len(states):
            raise ValueError('Cannot consolidate an empty class')
        states, queries = states.to(self.projection), queries.to(self.projection)
        flat = states.flatten(1)
        count = min(self.prototypes_per_class, len(states))
        # Deterministic farthest-point initialization followed by Lloyd updates.
        chosen = [int((flat - flat.mean(0)).square().sum(-1).argmax())]
        distance = (flat - flat[chosen[0]]).square().sum(-1)
        for _ in range(1, count):
            distance[chosen] = -1
            chosen.append(int(distance.argmax()))
            distance = torch.minimum(distance, (flat - flat[chosen[-1]]).square().sum(-1))
        centers = flat[chosen].clone()
        for _ in range(8):
            assignment = torch.cdist(flat, centers).argmin(-1)
            for cluster in range(count):
                member = assignment == cluster
                if member.any():
                    centers[cluster] = flat[member].mean(0)
        operators = self.routing_operators()
        slot = start
        for cluster in range(count):
            member = assignment == cluster
            if not member.any():
                continue
            density = states[member].mean(0)
            query = self._unit(queries[member].mean(0))
            probs, _, _ = self.route(query[None], density[None], operators)
            self.memory_states[slot].copy_(density)
            self.memory_queries[slot].copy_(query)
            self.memory_probs[slot, :, :probs.shape[-1]].copy_(probs[0])
            self.memory_counts[slot] = member.sum()
            self.memory_task[slot] = self.training_task
            self.memory_valid[slot] = True
            slot += 1


def add_compositional_args(parser):
    parser.add_argument('--comp_components_per_task', type=int, default=5)
    parser.add_argument('--comp_quantum_mix', type=float, default=0.5)
    parser.add_argument('--comp_prototypes_per_class', type=int, default=4)
    parser.add_argument('--comp_candidates_per_class', type=int, default=64)
    parser.add_argument('--comp_memory_batch_size', type=int, default=32)
    parser.add_argument('--comp_retention_coeff', type=float, default=1.0)
    parser.add_argument('--comp_prompt_coeff', type=float, default=1.0)
    parser.add_argument('--no_batchwise_prompt', action='store_false', dest='batchwise_prompt')


def validate_compositional_args(args):
    if args.prompt_router not in ('qsd_comp', 'cosine_comp'):
        return
    if args.distributed:
        raise ValueError('Compositional prompts currently support single-process training; run python main.py directly')
    if args.dataset != 'Split-CIFAR100':
        raise ValueError('This initial compositional preset supports Split-CIFAR100')
    if not args.prompt_pool or args.use_prompt_mask or args.shared_prompt_pool or args.shared_prompt_key:
        raise ValueError('Use prompt_pool=True without shared prompts, shared keys or task masks')
    if args.batchwise_prompt or args.task_inc:
        raise ValueError('Use --no_batchwise_prompt and class-incremental evaluation (task_inc=False)')
    if not args.reinit_optimizer:
        raise ValueError('Task-frozen banks require reinit_optimizer=True')
    if args.embedding_key != 'cls' or args.head_type != 'prompt':
        raise ValueError('Use frozen CLS queries and head_type=prompt')
    if not set(['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']).issubset(args.freeze):
        raise ValueError('Compositional prompts require the standard frozen ViT backbone')
    if args.comp_candidates_per_class < args.comp_prototypes_per_class:
        raise ValueError('Prototype candidate budget must cover prototypes_per_class')
    if min(args.comp_retention_coeff, args.comp_prompt_coeff) < 0 or args.epochs < 1:
        raise ValueError('Loss coefficients must be nonnegative and epochs positive')
    if args.qsd_no_cosine_prior:
        raise ValueError('For compositional prompts use --comp_quantum_mix 1 instead of --qsd_no_cosine_prior')
