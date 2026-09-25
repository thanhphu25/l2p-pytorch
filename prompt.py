import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumStateRouter(nn.Module):
    """Retrieve prompts by mixed-state quantum discrimination.

    Images and prompt keys become density matrices in a compact Hilbert space.
    A pretty-good measurement (PGM) converts their Born scores into prompt
    probabilities. One small, data-free density-state summary is retained per
    task so later measurements can be regularized without replaying images.
    """

    def __init__(self, pool_size, embed_dim, state_dim=16, rank=4, eps=1e-4,
                 cls_mix=0.5, cosine_tau=0.1, memory_size=10):
        super().__init__()
        if state_dim < 2:
            raise ValueError("qsd_state_dim must be at least 2")
        if rank < 1:
            raise ValueError("qsd_rank must be at least 1")
        if not 0.0 <= cls_mix <= 1.0:
            raise ValueError("qsd_cls_mix must be in [0, 1]")
        if cosine_tau <= 0.0:
            raise ValueError("qsd_cosine_tau must be positive")
        if memory_size < 1:
            raise ValueError("qsd_memory_size must be at least 1")

        self.pool_size = pool_size
        self.state_dim = state_dim
        self.rank = rank
        self.eps = eps
        self.cls_mix = cls_mix
        self.cosine_tau = cosine_tau
        self.memory_size = memory_size

        self.projection = nn.Linear(embed_dim, state_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

        # The projected key is the first factor. Extra factors let each prompt
        # represent a mixed state instead of only a rank-one pure state.
        if rank > 1:
            self.prompt_factors = nn.Parameter(
                torch.empty(pool_size, rank - 1, state_dim))
            nn.init.normal_(self.prompt_factors, std=0.02)
        else:
            self.register_parameter('prompt_factors', None)

        # Zero means the initial QSD top-k is exactly the cosine L2P top-k.
        self.residual_strength = nn.Parameter(torch.zeros(()))

        # O(tasks * d^2) state memory; it stores no images or ViT embeddings.
        self.register_buffer('anchor_states',
                             torch.zeros(memory_size, state_dim, state_dim))
        self.register_buffer('anchor_probs',
                             torch.zeros(memory_size, pool_size))
        self.register_buffer('anchor_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('pending_state_sum',
                             torch.zeros(state_dim, state_dim))
        self.register_buffer('pending_prob_sum', torch.zeros(pool_size))
        self.register_buffer('pending_count', torch.zeros((), dtype=torch.long))

    @staticmethod
    def _trace_normalize(states, eps):
        states = 0.5 * (states + states.transpose(-1, -2))
        trace = states.diagonal(dim1=-2, dim2=-1).sum(-1)
        return states / trace.clamp_min(eps)[..., None, None]

    def prompt_states(self, prompt_keys):
        base = F.normalize(self.projection(prompt_keys.float()), dim=-1).unsqueeze(1)
        if self.prompt_factors is not None:
            extra = F.normalize(self.prompt_factors.float(), dim=-1)
            factors = torch.cat((base, extra), dim=1)
        else:
            factors = base
        states = torch.einsum('mrd,mre->mde', factors, factors)
        return self._trace_normalize(states, self.eps)

    def image_states(self, queries, patch_tokens=None):
        query_factors = F.normalize(self.projection(queries.float()), dim=-1)
        cls_states = torch.einsum('bd,be->bde', query_factors, query_factors)
        if patch_tokens is None or self.cls_mix == 1.0:
            return cls_states

        patch_factors = F.normalize(self.projection(patch_tokens.float()), dim=-1)
        attention = torch.einsum(
            'bnd,bd->bn', patch_factors, query_factors) / math.sqrt(self.state_dim)
        attention = attention.softmax(dim=1)
        patch_states = torch.einsum(
            'bn,bnd,bne->bde', attention, patch_factors, patch_factors)
        states = self.cls_mix * cls_states + (1.0 - self.cls_mix) * patch_states
        return self._trace_normalize(states, self.eps)

    def pretty_good_measurement(self, prompt_states):
        prior_states = prompt_states / self.pool_size
        ensemble = prior_states.sum(dim=0)
        eye = torch.eye(
            self.state_dim, device=ensemble.device, dtype=ensemble.dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(ensemble + self.eps * eye)
        inverse_sqrt = eigenvectors @ torch.diag_embed(
            eigenvalues.clamp_min(self.eps).rsqrt()) @ eigenvectors.transpose(-1, -2)
        measurement = inverse_sqrt.unsqueeze(0) @ prior_states @ inverse_sqrt.unsqueeze(0)
        return 0.5 * (measurement + measurement.transpose(-1, -2))

    def measure(self, measurement, image_states):
        probabilities = torch.einsum('mde,bed->bm', measurement, image_states)
        probabilities = probabilities.clamp_min(self.eps)
        return probabilities / probabilities.sum(dim=1, keepdim=True)

    def retention_loss(self, measurement):
        count = int(self.anchor_count.item())
        if count == 0:
            return measurement.new_zeros(())
        current_probs = self.measure(measurement, self.anchor_states[:count])
        target_probs = self.anchor_probs[:count].clamp_min(self.eps)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
        return F.kl_div(current_probs.log(), target_probs, reduction='batchmean')

    @torch.no_grad()
    def _accumulate_memory(self, states, probabilities):
        self.pending_state_sum.add_(states.detach().sum(dim=0))
        self.pending_prob_sum.add_(probabilities.detach().sum(dim=0))
        self.pending_count.add_(states.shape[0])

    @torch.no_grad()
    def consolidate_task(self):
        """Commit the current task summary to the data-free state memory."""
        count = int(self.pending_count.item())
        if count == 0:
            return

        state = self._trace_normalize(self.pending_state_sum / count, self.eps)
        probability = self.pending_prob_sum / count
        probability = probability / probability.sum().clamp_min(self.eps)

        num_anchors = int(self.anchor_count.item())
        if num_anchors < self.memory_size:
            index = num_anchors
            self.anchor_count.add_(1)
        else:
            self.anchor_states[:-1].copy_(self.anchor_states[1:].clone())
            self.anchor_probs[:-1].copy_(self.anchor_probs[1:].clone())
            index = self.memory_size - 1

        self.anchor_states[index].copy_(state)
        self.anchor_probs[index].copy_(probability)
        self.pending_state_sum.zero_()
        self.pending_prob_sum.zero_()
        self.pending_count.zero_()

    def forward(self, queries, patch_tokens, prompt_keys, update_memory=False):
        # Eigh is substantially more stable in fp32 than under mixed precision.
        prompt_states = self.prompt_states(prompt_keys)
        image_states = self.image_states(queries, patch_tokens)
        measurement = self.pretty_good_measurement(prompt_states)
        probabilities = self.measure(measurement, image_states)

        if update_memory:
            self._accumulate_memory(image_states, probabilities)

        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(1).mean()
        purity = torch.einsum('bde,bed->b', image_states, image_states).mean()
        return {
            'probabilities': probabilities,
            'retention_loss': self.retention_loss(measurement),
            'entropy': entropy,
            'purity': purity,
        }


class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 prompt_router='cosine', qsd_state_dim=16, qsd_rank=4, qsd_eps=1e-4, qsd_cls_mix=0.5,
                 qsd_cosine_tau=0.1, qsd_memory_size=10, qsd_no_cosine_prior=False):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.prompt_router = prompt_router
        self.qsd_no_cosine_prior = qsd_no_cosine_prior

        if prompt_router not in ('cosine', 'qsd'):
            raise ValueError("prompt_router must be 'cosine' or 'qsd'")

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # Retain the original L2P behavior for the baseline path.
            self.prompt_key = torch.mean(self.prompt, dim=1)

        if prompt_router == 'qsd':
            if not prompt_pool or not prompt_key:
                raise ValueError("QSD routing requires a prompt pool and learnable prompt keys")
            self.quantum_router = QuantumStateRouter(
                pool_size=pool_size,
                embed_dim=embed_dim,
                state_dim=qsd_state_dim,
                rank=qsd_rank,
                eps=qsd_eps,
                cls_mix=qsd_cls_mix,
                cosine_tau=qsd_cosine_tau,
                memory_size=qsd_memory_size,
            )

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(
            square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    @torch.no_grad()
    def consolidate_router(self):
        if hasattr(self, 'quantum_router'):
            self.quantum_router.consolidate_task()

    def forward(self, x_embed, prompt_mask=None, cls_features=None,
                prompt_query_tokens=None, update_router_memory=False):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                x_embed_mean = (torch.max(x_embed, dim=1)[0]
                                if cls_features is None else cls_features)
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            selection_scores = similarity
            route_probabilities = None

            if hasattr(self, 'quantum_router'):
                quantum = self.quantum_router(
                    x_embed_mean, prompt_query_tokens, self.prompt_key,
                    update_memory=update_router_memory)
                quantum_log_prob = quantum['probabilities'].clamp_min(
                    self.quantum_router.eps).log()
                if self.qsd_no_cosine_prior:
                    routing_logits = quantum_log_prob
                else:
                    routing_logits = (
                        similarity / self.quantum_router.cosine_tau
                        + self.quantum_router.residual_strength * quantum_log_prob)
                route_probabilities = routing_logits.softmax(dim=1)
                selection_scores = routing_logits

                out['qsd_probabilities'] = quantum['probabilities']
                out['qsd_retention_loss'] = quantum['retention_loss']
                out['qsd_entropy'] = quantum['entropy']
                out['qsd_purity'] = quantum['purity']
                out['qsd_strength'] = self.quantum_router.residual_strength
                out['route_entropy'] = -(
                    route_probabilities
                    * route_probabilities.clamp_min(self.quantum_router.eps).log()
                ).sum(1).mean()

            if prompt_mask is None:
                _, idx = torch.topk(selection_scores, k=self.top_k, dim=1)
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full(
                            (self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()),
                            device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full(
                            (self.pool_size - id_counts.shape[0],), 0,
                            device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)
                    major_prompt_id = prompt_id[major_idx]
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)
            else:
                idx = prompt_mask

            batched_prompt_raw = self.prompt[idx]

            # Differentiable routing, but exactly unchanged prompt magnitude in
            # the forward pass (straight-through estimator).
            if route_probabilities is not None:
                selected_probability = route_probabilities.gather(1, idx)
                straight_through_scale = (
                    1.0 + selected_probability - selected_probability.detach())
                batched_prompt_raw = (
                    batched_prompt_raw * straight_through_scale[..., None, None])

            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

            out['prompt_idx'] = idx
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity
            if route_probabilities is not None:
                out['route_probabilities'] = route_probabilities

            # Preserve the original L2P pull-constraint for a fair ablation.
            batched_key_norm = prompt_norm[idx]
            out['selected_key'] = batched_key_norm
            query_norm = x_embed_norm.unsqueeze(1)
            sim = batched_key_norm * query_norm
            out['reduce_sim'] = torch.sum(sim) / x_embed.shape[0]
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        return out
