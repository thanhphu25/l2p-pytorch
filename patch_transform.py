import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricFourQubitCircuit(nn.Module):
    """Differentiable 4-qubit RY/RZ+CNOT circuit.

    Real and imaginary amplitudes are stored separately for compatibility with
    the PyTorch 1.12 CUDA environment used by the original L2P repository.
    """

    def __init__(self, depth=2, init_std=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError('phase_circuit_depth must be at least 1')
        if init_std < 0.0:
            raise ValueError('phase_circuit_init_std must be non-negative')

        self.num_qubits = 4
        self.num_states = 2 ** self.num_qubits
        self.depth = depth
        self.angles = nn.Parameter(torch.empty(depth, self.num_qubits, 2))
        nn.init.normal_(self.angles, std=init_std)

        permutations = []
        for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
            control_bit = 1 << (self.num_qubits - 1 - control)
            target_bit = 1 << (self.num_qubits - 1 - target)
            permutation = []
            for index in range(self.num_states):
                source = index ^ target_bit if index & control_bit else index
                permutation.append(source)
            permutations.append(permutation)
        self.register_buffer(
            'cnot_permutations', torch.tensor(permutations, dtype=torch.long))

    def _apply_ry(self, component, angle, qubit):
        cosine = torch.cos(angle / 2.0)
        sine = torch.sin(angle / 2.0)
        gate = torch.stack((
            torch.stack((cosine, -sine)),
            torch.stack((sine, cosine)),
        ))
        shaped = component.reshape(-1, 2, 2, 2, 2).movedim(qubit + 1, -1)
        shaped = torch.matmul(shaped, gate.transpose(0, 1))
        return shaped.movedim(-1, qubit + 1).reshape(-1, self.num_states)

    def _apply_rz(self, real, imaginary, angle, qubit):
        phase = torch.stack((-angle / 2.0, angle / 2.0))
        cosine = torch.cos(phase)
        sine = torch.sin(phase)
        real_shaped = real.reshape(-1, 2, 2, 2, 2).movedim(qubit + 1, -1)
        imag_shaped = imaginary.reshape(-1, 2, 2, 2, 2).movedim(qubit + 1, -1)
        rotated_real = real_shaped * cosine - imag_shaped * sine
        rotated_imag = real_shaped * sine + imag_shaped * cosine
        rotated_real = rotated_real.movedim(-1, qubit + 1).reshape(-1, self.num_states)
        rotated_imag = rotated_imag.movedim(-1, qubit + 1).reshape(-1, self.num_states)
        return rotated_real, rotated_imag

    def forward(self, phases):
        if phases.shape[-1] != self.num_states:
            raise ValueError('The 4-qubit circuit expects exactly 16 phases')

        amplitude = 1.0 / math.sqrt(self.num_states)
        real = amplitude * torch.cos(phases)
        imaginary = amplitude * torch.sin(phases)

        for layer in range(self.depth):
            for qubit in range(self.num_qubits):
                ry_angle, rz_angle = self.angles[layer, qubit]
                real = self._apply_ry(real, ry_angle, qubit)
                imaginary = self._apply_ry(imaginary, ry_angle, qubit)
                real, imaginary = self._apply_rz(
                    real, imaginary, rz_angle, qubit)
            for permutation in self.cnot_permutations:
                real = real.index_select(1, permutation)
                imaginary = imaginary.index_select(1, permutation)

        probabilities = real.square() + imaginary.square()
        return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


class PromptConditionedPatchTransform(nn.Module):
    """Patch reweighting controls used on top of otherwise original L2P."""

    MODES = ('phase', 'mlp', 'phase_no_encoding')

    def __init__(self, embed_dim, mode='phase', latent_dim=32,
                 circuit_depth=2, alpha_init=0.05, alpha_max=0.2,
                 circuit_init_std=0.1):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError('Unsupported patch transform: {}'.format(mode))
        if latent_dim < 1:
            raise ValueError('phase_latent_dim must be positive')
        if not 0.0 < alpha_init < alpha_max:
            raise ValueError('Require 0 < phase_alpha_init < phase_alpha_max')

        self.mode = mode
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.grid_size = 4
        self.num_regions = self.grid_size ** 2
        self.alpha_max = alpha_max

        alpha_ratio = alpha_init / alpha_max
        self.alpha_logit = nn.Parameter(torch.tensor(
            math.log(alpha_ratio / (1.0 - alpha_ratio))))

        if mode in ('phase', 'phase_no_encoding'):
            self.patch_phase_projection = nn.Linear(
                embed_dim, latent_dim, bias=False)
            self.prompt_phase_projection = nn.Linear(
                embed_dim, latent_dim, bias=False)
            nn.init.xavier_uniform_(self.patch_phase_projection.weight)
            nn.init.xavier_uniform_(self.prompt_phase_projection.weight)
            self.circuit = ParametricFourQubitCircuit(
                depth=circuit_depth, init_std=circuit_init_std)
        else:
            # At ViT-B/16, latent_dim=32: 49,218 versus 49,169 parameters
            # for phase (<0.1% difference).
            self.patch_mlp = nn.Sequential(
                nn.Linear(2 * embed_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_logit)

    def _pool_to_regions(self, patch_embeddings):
        batch_size, num_patches, channels = patch_embeddings.shape
        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError('Patch interference requires a square patch grid')
        patch_grid = patch_embeddings.transpose(1, 2).reshape(
            batch_size, channels, side, side)
        pooled = F.adaptive_avg_pool2d(
            patch_grid, (self.grid_size, self.grid_size))
        return pooled.flatten(2).transpose(1, 2), side

    def _phase_probabilities(self, regions, prompt_summary):
        normalized_regions = F.layer_norm(regions, (self.embed_dim,))
        normalized_prompt = F.layer_norm(prompt_summary, (self.embed_dim,))
        patch_factors = self.patch_phase_projection(normalized_regions)
        prompt_factor = self.prompt_phase_projection(normalized_prompt)
        phase_logits = torch.einsum(
            'brd,bd->br', patch_factors, prompt_factor)
        phase_logits = phase_logits / math.sqrt(self.latent_dim)
        if self.mode == 'phase_no_encoding':
            # Preserve a zero-gradient graph edge for DDP parameter accounting.
            phases = phase_logits * 0.0
        else:
            phases = math.pi * torch.tanh(phase_logits)
        return self.circuit(phases), phases

    def _mlp_probabilities(self, regions, prompt_summary):
        normalized_regions = F.layer_norm(regions, (self.embed_dim,))
        normalized_prompt = F.layer_norm(prompt_summary, (self.embed_dim,))
        expanded_prompt = normalized_prompt.unsqueeze(1).expand(
            -1, self.num_regions, -1)
        logits = self.patch_mlp(
            torch.cat((normalized_regions, expanded_prompt), dim=-1)).squeeze(-1)
        return logits.softmax(dim=1), logits.new_zeros(logits.shape)

    def forward(self, patch_embeddings, selected_prompts):
        regions, patch_side = self._pool_to_regions(patch_embeddings)
        prompt_summary = selected_prompts.mean(dim=(1, 2))
        if self.mode in ('phase', 'phase_no_encoding'):
            probabilities, phases = self._phase_probabilities(
                regions, prompt_summary)
        else:
            probabilities, phases = self._mlp_probabilities(
                regions, prompt_summary)

        probability_map = probabilities.reshape(
            -1, 1, self.grid_size, self.grid_size)
        probability_map = F.interpolate(
            probability_map, size=(patch_side, patch_side),
            mode='bilinear', align_corners=False)
        probability_map = probability_map.flatten(2).transpose(1, 2)
        scale = 1.0 + self.alpha * torch.tanh(
            self.num_regions * probability_map - 1.0)
        modulated = patch_embeddings * scale

        entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1).mean() / math.log(self.num_regions)
        diagnostics = {
            'patch_probabilities': probabilities,
            'patch_entropy': entropy,
            'patch_deviation': (scale - 1.0).abs().mean(),
            'patch_alpha': self.alpha,
            'patch_peak': probabilities.max(dim=1)[0].mean(),
            'phase_std': phases.std(dim=1, unbiased=False).mean(),
        }
        return modulated, diagnostics
