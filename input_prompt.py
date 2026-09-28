"""Input-space prompt pool with classical and quantum-inspired routers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_one_qubit_gate(state, gate, qubit, num_qubits):
    """Apply a batched 2x2 gate to one axis of a batched state vector."""
    batch = state.shape[0]
    state = state.reshape([batch] + [2] * num_qubits)
    axis = qubit + 1
    permutation = [0, axis] + [i for i in range(1, num_qubits + 1) if i != axis]
    inverse = [permutation.index(i) for i in range(num_qubits + 1)]
    state = state.permute(permutation).reshape(batch, 2, -1)
    state = torch.einsum('bij,bjk->bik', gate, state)
    return state.reshape([batch, 2] + [2] * (num_qubits - 1)).permute(inverse).reshape(batch, -1)


class QuantumRouter(nn.Module):
    """Differentiable state-vector circuit used only as a prompt router.

    This is quantum-inspired simulation, not execution on quantum hardware.
    Similarities define amplitudes, a query projection defines phases, and
    trainable rotations/interference produce prompt-selection probabilities.
    """

    def __init__(self, query_dim, pool_size, top_k, depth=2, temperature=1.0,
                 use_data_phase=True):
        super().__init__()
        if top_k < 2:
            raise ValueError('quantum routing requires input_prompt_top_k >= 2')
        if depth < 1:
            raise ValueError('input_prompt_quantum_depth must be >= 1')
        if temperature <= 0:
            raise ValueError('input_prompt_temperature must be > 0')
        self.top_k = top_k
        self.num_qubits = int(math.ceil(math.log2(top_k)))
        self.state_dim = 2 ** self.num_qubits
        self.depth = depth
        self.temperature = temperature
        self.use_data_phase = use_data_phase

        self.phase_encoder = nn.Linear(query_dim, pool_size)
        self.ry = nn.Parameter(torch.zeros(depth, self.num_qubits))
        self.rz = nn.Parameter(torch.zeros(depth, self.num_qubits))
        self.entangle = nn.Parameter(torch.zeros(depth, self.num_qubits))
        self.measure_ry = nn.Parameter(torch.zeros(self.num_qubits))
        nn.init.normal_(self.ry, std=0.02)
        nn.init.normal_(self.rz, std=0.02)
        nn.init.normal_(self.measure_ry, std=0.02)

    @staticmethod
    def _ry_gate(theta, batch):
        half = theta / 2
        row0 = torch.stack((torch.cos(half), -torch.sin(half)))
        row1 = torch.stack((torch.sin(half), torch.cos(half)))
        return torch.stack((row0, row1)).to(torch.complex64).unsqueeze(0).expand(batch, -1, -1)

    @staticmethod
    def _rz_gate(theta, batch):
        half = theta / 2
        zero = torch.zeros_like(half)
        diag0 = torch.polar(torch.ones_like(half), -half)
        diag1 = torch.polar(torch.ones_like(half), half)
        row0 = torch.stack((diag0, zero.to(diag0.dtype)))
        row1 = torch.stack((zero.to(diag1.dtype), diag1))
        return torch.stack((row0, row1)).unsqueeze(0).expand(batch, -1, -1)

    def _controlled_phase(self, state, theta, qubit):
        indices = torch.arange(self.state_dim, device=state.device)
        target = (qubit + 1) % self.num_qubits
        active = (((indices >> qubit) & 1) * ((indices >> target) & 1)).to(state.real.dtype)
        phase = torch.polar(torch.ones_like(active), active * theta)
        return state * phase.unsqueeze(0)

    def forward(self, query, selected_scores, selected_idx):
        batch = query.shape[0]
        probabilities = F.softmax(selected_scores / self.temperature, dim=-1)
        amplitudes = torch.sqrt(probabilities.clamp_min(1e-8))

        phase = self.phase_encoder(query).gather(1, selected_idx)
        if not self.use_data_phase:
            phase = torch.zeros_like(phase)
        state = torch.polar(amplitudes, math.pi * torch.tanh(phase))
        if self.state_dim > self.top_k:
            state = F.pad(state, (0, self.state_dim - self.top_k))

        for layer in range(self.depth):
            for qubit in range(self.num_qubits):
                state = _apply_one_qubit_gate(
                    state, self._ry_gate(self.ry[layer, qubit], batch), qubit, self.num_qubits)
                state = _apply_one_qubit_gate(
                    state, self._rz_gate(self.rz[layer, qubit], batch), qubit, self.num_qubits)
            for qubit in range(self.num_qubits):
                state = self._controlled_phase(state, self.entangle[layer, qubit], qubit)
        # Convert relative phase back into measurement probability.
        for qubit in range(self.num_qubits):
            state = _apply_one_qubit_gate(
                state, self._ry_gate(self.measure_ry[qubit], batch), qubit, self.num_qubits)

        weights = state.abs().square()[:, :self.top_k]
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class InputSpatialPrompt(nn.Module):
    """LGSP-style local/global input prompting with controlled router ablations."""

    ROUTERS = {'cosine', 'linear', 'quantum', 'quantum_no_phase'}

    def __init__(self, query_dim, pool_size=10, top_k=5, hidden_dim=8,
                 router='cosine', global_prompt=False, frequency_rings=8,
                 max_scale=0.1, init_scale=0.01, quantum_depth=2,
                 temperature=1.0):
        super().__init__()
        if router not in self.ROUTERS:
            raise ValueError('unknown input prompt router: {}'.format(router))
        if not 1 <= top_k <= pool_size:
            raise ValueError('input_prompt_top_k must be in [1, input_prompt_pool_size]')
        if frequency_rings < 1:
            raise ValueError('input_prompt_frequency_rings must be >= 1')
        if temperature <= 0:
            raise ValueError('input_prompt_temperature must be > 0')
        if max_scale <= 0:
            raise ValueError('input_prompt_max_scale must be > 0')
        if not 0 < init_scale < max_scale:
            raise ValueError('input_prompt_init_scale must be between 0 and input_prompt_max_scale')

        self.router_name = router
        self.pool_size = pool_size
        self.top_k = top_k
        self.temperature = temperature
        self.global_prompt = global_prompt
        self.frequency_rings = frequency_rings
        self.max_scale = max_scale

        # Generate prompt masks on the ViT patch grid, then upsample.
        self.local_generator = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, pool_size, kernel_size=3, padding=1),
        )
        self.prompt_color = nn.Parameter(torch.empty(pool_size, 3))
        nn.init.normal_(self.prompt_color, std=0.02)
        self.prompt_key = nn.Parameter(torch.empty(pool_size, query_dim))
        nn.init.uniform_(self.prompt_key, -1, 1)

        if router == 'linear':
            self.linear_router = nn.Linear(query_dim, pool_size)
        elif router in {'quantum', 'quantum_no_phase'}:
            self.quantum_router = QuantumRouter(
                query_dim, pool_size, top_k, depth=quantum_depth,
                temperature=temperature, use_data_phase=router == 'quantum')

        if global_prompt:
            self.frequency_logits = nn.Parameter(torch.zeros(frequency_rings))

        init_logit = math.log(init_scale / (max_scale - init_scale))
        self.local_scale_logit = nn.Parameter(torch.tensor(init_logit))
        if global_prompt:
            self.global_scale_logit = nn.Parameter(torch.tensor(init_logit))

    @staticmethod
    def _normalize(x):
        return F.normalize(x, dim=-1, eps=1e-12)

    def _route(self, query):
        query_norm = self._normalize(query)
        key_norm = self._normalize(self.prompt_key)
        similarity = query_norm @ key_norm.t()
        selected_scores, selected_idx = torch.topk(similarity, self.top_k, dim=-1)

        if self.router_name == 'cosine':
            weights = F.softmax(selected_scores / self.temperature, dim=-1)
        elif self.router_name == 'linear':
            learned_logits = self.linear_router(query_norm).gather(1, selected_idx)
            weights = F.softmax(selected_scores / self.temperature + learned_logits, dim=-1)
        else:
            weights = self.quantum_router(query_norm, selected_scores, selected_idx)
        return selected_idx, weights, similarity

    def _local_prompt(self, x, selected_idx, weights):
        patch_grid = max(4, min(x.shape[-2:]) // 16)
        low_res = F.adaptive_avg_pool2d(x, (patch_grid, patch_grid))
        masks = torch.tanh(self.local_generator(low_res))
        gather_idx = selected_idx[:, :, None, None].expand(-1, -1, patch_grid, patch_grid)
        masks = masks.gather(1, gather_idx)
        colors = self.prompt_color[selected_idx]
        prompt = torch.einsum('bk,bkhw,bkc->bchw', weights, masks, colors)
        return F.interpolate(prompt, size=x.shape[-2:], mode='bilinear', align_corners=False)

    def _frequency_prompt(self, x):
        height, width = x.shape[-2:]
        yy = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())
        radius = radius / radius.max().clamp_min(1e-8)
        ring_id = torch.clamp((radius * self.frequency_rings).long(), max=self.frequency_rings - 1)

        # Uniform initialization is all-pass and therefore an exact zero residual.
        ring_weights = F.softmax(self.frequency_logits, dim=0) * self.frequency_rings
        frequency_mask = ring_weights[ring_id]
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
        filtered = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * frequency_mask[None, None], dim=(-2, -1)),
            norm='ortho').real
        return filtered - x

    def forward(self, x, query):
        if query is None:
            raise ValueError('input prompting requires cls_features from the frozen query model')
        selected_idx, weights, similarity = self._route(query)
        local = self._local_prompt(x, selected_idx, weights)
        local_scale = self.max_scale * torch.sigmoid(self.local_scale_logit)
        prompted = x + local_scale * local

        global_scale = x.new_zeros(())
        if self.global_prompt:
            global_scale = self.max_scale * torch.sigmoid(self.global_scale_logit)
            prompted = prompted + global_scale * self._frequency_prompt(x)

        diagnostics = {
            'input_prompt_idx': selected_idx,
            'input_prompt_weights': weights,
            'input_prompt_similarity': similarity,
            'input_prompt_local_scale': local_scale,
            'input_prompt_global_scale': global_scale,
        }
        return prompted, diagnostics
