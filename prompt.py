import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxGate(nn.Module):
    """Parameter-free gate: w_i = softmax(s_i / tau) over the selected top-k prompts."""
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, sim, idx, query):
        return F.softmax(sim / self.tau, dim=1)


class LinearGate(nn.Module):
    """Classical control with the same parameter budget as QuantumGate:
    w_i = softmax(s_i / tau + <u_{j_i}, q(x)> + b_{j_i}), where j_i is the pool index of the i-th selected prompt.
    Zero init, so it starts exactly as SoftmaxGate.
    """
    def __init__(self, pool_size, embed_dim, tau=0.1):
        super().__init__()
        self.tau = tau
        self.weight = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.bias = nn.Parameter(torch.zeros(pool_size))

    def forward(self, sim, idx, query):
        logit = (query @ self.weight.t() + self.bias).gather(1, idx) # B, top_k
        return F.softmax(sim / self.tau + logit, dim=1)


class QuantumGate(nn.Module):
    """Quantum interference prompt gating, simulated exactly as a 2^n_qubits state vector.

    a_i(x) = sqrt(softmax(s_i / tau)) * exp(i * phi_{j_i}(x)),  i = 1..top_k (remaining amplitudes are 0)
    w_i(x) = |[U_theta a(x)]_i|^2 / sum_{j <= top_k} |[U_theta a(x)]_j|^2

    phi_j(x) = <u_j, q(x)> + b_j is a per-prompt phase read from the query feature.
    Each circuit layer is E^† R_b E R_a, where R_* are per-qubit ZYZ rotations and E is the CNOT ladder
    CNOT(0->1) ... CNOT(n-2->n-1). The ladder is mirrored so that U_theta = I at theta = 0: together with
    zero-initialised phases the gate starts exactly as SoftmaxGate, and entanglement grows as theta is learned.
    """
    def __init__(self, top_k, pool_size, embed_dim, tau=0.1, n_qubits=3, n_layers=2):
        super().__init__()
        assert 2 ** n_qubits >= top_k, f'{n_qubits} qubits cannot hold {top_k} prompts'
        self.tau = tau
        self.top_k = top_k
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

        self.phase_weight = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.phase_bias = nn.Parameter(torch.zeros(pool_size))
        # theta[layer, block (a: before the ladder, b: inside it), qubit, ZYZ euler angles]
        self.theta = nn.Parameter(torch.zeros(n_layers, 2, n_qubits, 3))
        self.register_buffer('ladder', self._cnot_ladder(n_qubits), persistent=False)

    @staticmethod
    def _cnot_ladder(n_qubits):
        # basis index i = sum_q b_q * 2^(n-1-q), i.e. qubit 0 is the most significant bit
        dim = 2 ** n_qubits
        ladder = torch.eye(dim)
        for control in range(n_qubits - 1):
            c_bit, t_bit = 1 << (n_qubits - 1 - control), 1 << (n_qubits - 2 - control)
            perm = torch.zeros(dim, dim)
            for i in range(dim):
                perm[i ^ t_bit if i & c_bit else i, i] = 1.
            ladder = perm @ ladder
        return ladder

    @staticmethod
    def _rotations(angles):
        """Kronecker product over qubits of RZ(gamma) RY(beta) RZ(alpha); angles: (n_qubits, 3)."""
        alpha, beta, gamma = angles.unbind(-1)
        c, s = torch.cos(beta / 2), torch.sin(beta / 2)
        e_sum, e_diff = torch.exp(-0.5j * (alpha + gamma)), torch.exp(0.5j * (alpha - gamma))
        rot = torch.stack([
            torch.stack([e_sum * c, -e_diff * s], dim=-1),
            torch.stack([e_diff.conj() * s, e_sum.conj() * c], dim=-1),
        ], dim=-2) # n_qubits, 2, 2
        full = rot[0]
        for r in rot[1:]:
            full = torch.einsum('ab,cd->acbd', full, r).reshape(full.shape[0] * 2, full.shape[1] * 2)
        return full

    def unitary(self):
        ladder = self.ladder.to(torch.complex64)
        u = torch.eye(self.dim, dtype=torch.complex64, device=self.theta.device)
        for layer in self.theta.float():
            u = ladder.t() @ self._rotations(layer[1]) @ ladder @ self._rotations(layer[0]) @ u
        return u

    def forward(self, sim, idx, query):
        with torch.autocast(device_type=sim.device.type, enabled=False):
            sim, query = sim.float(), query.float()
            magnitude = torch.sqrt(F.softmax(sim / self.tau, dim=1)) # B, top_k
            phase = (query @ self.phase_weight.t() + self.phase_bias).gather(1, idx) # B, top_k
            amp = F.pad(torch.polar(magnitude, phase), (0, self.dim - self.top_k)) # B, 2^n
            state = (amp @ self.unitary().t())[:, :self.top_k]
            prob = state.real ** 2 + state.imag ** 2 # |.|^2 without abs(), whose gradient is undefined at 0
            return prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-12)


class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 gating='none', gate_tau=0.1, gate_qubits=3, gate_layers=2, gate_train_sim=False,):
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

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
        # optional reweighting of the selected top-k prompts: P_i -> top_k * w_i * P_i
        self.gating = gating
        self.gate_tau = gate_tau
        self.gate_train_sim = gate_train_sim
        if gating == 'none':
            self.gate = None
        else:
            assert prompt_pool, 'prompt gating needs a prompt pool'
            if gating == 'softmax':
                self.gate = SoftmaxGate(tau=gate_tau)
            elif gating == 'linear':
                self.gate = LinearGate(pool_size, embed_dim, tau=gate_tau)
            elif gating == 'quantum':
                self.gate = QuantumGate(top_k, pool_size, embed_dim, tau=gate_tau, n_qubits=gate_qubits, n_layers=gate_layers)
            else:
                raise NotImplementedError(f'Not supported prompt gating: {gating}')

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape

            if self.gate is not None:
                selected_sim = similarity.gather(1, idx) # B, top_k
                if not self.gate_train_sim:
                    # keep L2P's decoupling: keys are only trained by the pull constraint, not by the CE loss
                    selected_sim = selected_sim.detach()
                gate_weights = self.gate(selected_sim, idx, x_embed_norm) # B, top_k, rows sum to 1
                batched_prompt_raw = batched_prompt_raw * (top_k * gate_weights).to(batched_prompt_raw.dtype)[:, :, None, None]
                out['gate_weights'] = gate_weights
                out['gate_prior'] = F.softmax(selected_sim.detach() / self.gate_tau, dim=1)

            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
