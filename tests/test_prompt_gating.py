"""Sanity checks for the prompt gates. Run from the repo root: python tests/test_prompt_gating.py"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt import LinearGate, Prompt, QuantumGate, SoftmaxGate


def reference_unitary(theta):
    """Independent numpy construction of the circuit: textbook CNOT via projectors and np.kron."""
    n_layers, _, n_qubits, _ = theta.shape
    I, X = np.eye(2), np.array([[0, 1], [1, 0]])
    P0, P1 = np.diag([1, 0]), np.diag([0, 1])

    def on_qubits(ops):
        out = np.eye(1)
        for op in ops:
            out = np.kron(out, op)
        return out

    def cnot(control, target):
        a = [I] * n_qubits
        b = [I] * n_qubits
        a[control], b[control], b[target] = P0, P1, X
        return on_qubits(a) + on_qubits(b)

    def rz(t):
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    def ry(t):
        return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])

    def rotations(angles):
        return on_qubits([rz(g) @ ry(b) @ rz(a) for a, b, g in angles])

    ladder = np.eye(2 ** n_qubits)
    for q in range(n_qubits - 1):
        ladder = cnot(q, q + 1) @ ladder
    u = np.eye(2 ** n_qubits)
    for layer in theta:
        u = ladder.conj().T @ rotations(layer[1]) @ ladder @ rotations(layer[0]) @ u
    return u


def random_inputs(batch=6, pool_size=10, top_k=5, embed_dim=16):
    sim = torch.rand(batch, top_k) * 2 - 1
    idx = torch.stack([torch.randperm(pool_size)[:top_k] for _ in range(batch)])
    query = F.normalize(torch.randn(batch, embed_dim), dim=1)
    return sim, idx, query


def test_circuit_matches_reference():
    gate = QuantumGate(top_k=5, pool_size=10, embed_dim=16, n_qubits=3, n_layers=2)
    with torch.no_grad():
        gate.theta.normal_(0, 1.5)
    u = gate.unitary().detach().numpy()
    assert np.allclose(u, reference_unitary(gate.theta.detach().double().numpy()), atol=1e-5)
    assert np.allclose(u.conj().T @ u, np.eye(8), atol=1e-5)
    # ladder CNOT(0->1) then CNOT(1->2): |100> -> |110> -> |111>, |010> -> |011>
    assert gate.ladder[7, 4] == 1 and gate.ladder[3, 2] == 1


def test_gates_start_as_softmax():
    sim, idx, query = random_inputs()
    prior = F.softmax(sim / 0.1, dim=1)
    for gate in [SoftmaxGate(0.1), LinearGate(10, 16, 0.1), QuantumGate(5, 10, 16, 0.1)]:
        assert torch.allclose(gate(sim, idx, query), prior, atol=1e-6), type(gate).__name__


def test_quantum_weights_and_interference():
    sim, idx, query = random_inputs()
    gate = QuantumGate(5, 10, 16, 0.1)
    with torch.no_grad():
        gate.theta.normal_(0, 1.0)
    w = gate(sim, idx, query)
    assert (w >= 0).all() and torch.allclose(w.sum(1), torch.ones(len(w)), atol=1e-5)
    with torch.no_grad():
        gate.phase_bias.normal_(0, 1.0)
    # once the circuit mixes amplitudes, relative phases change the measured weights
    assert not torch.allclose(gate(sim, idx, query), w, atol=1e-3)


def test_quantum_gradients():
    sim, idx, query = random_inputs()
    gate = QuantumGate(5, 10, 16, 0.1)
    target = torch.rand(5)
    # at init (U = I) the RY angles already get a gradient, so learning can leave the softmax point
    (gate(sim, idx, query) * target).sum().backward()
    assert gate.theta.grad[..., 1].abs().sum() > 0
    gate.zero_grad()
    with torch.no_grad():
        gate.theta.normal_(0, 0.5)
    (gate(sim, idx, query) * target).sum().backward()
    for name, p in gate.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_prompt_module():
    torch.manual_seed(0)
    x_embed, cls = torch.randn(4, 7, 16), torch.randn(4, 16)
    outputs = {}
    for gating in ['none', 'softmax', 'linear', 'quantum']:
        torch.manual_seed(1)
        prompt = Prompt(length=3, embed_dim=16, embedding_key='cls', prompt_pool=True, prompt_key=True, pool_size=10,
                        top_k=5, batchwise_prompt=False, gating=gating, gate_tau=1e6)
        out = prompt(x_embed, cls_features=cls)
        assert out['prompted_embedding'].shape == (4, 5 * 3 + 7, 16)
        outputs[gating] = out['prompted_embedding']
        if gating != 'none':
            assert torch.allclose(out['gate_weights'].sum(1), torch.ones(4), atol=1e-5)
            # the CE loss must not reach the keys through the gate unless --gate_train_sim is set
            out['prompted_embedding'].sum().backward()
            assert prompt.prompt_key.grad is None or prompt.prompt_key.grad.abs().sum() == 0
    # with a huge temperature every gate is uniform, so 5 * w_i = 1 and prompts are left unchanged
    for gating in ['softmax', 'linear', 'quantum']:
        assert torch.allclose(outputs[gating], outputs['none'], atol=1e-5), gating


def test_train_and_evaluate_end_to_end():
    """Two tiny tasks through the real engine with a tiny random ViT, then check summary.json."""
    from timm.models import create_model
    from configs.cifar100_l2p import get_args_parser
    import models  # noqa: F401  (registers the prompt ViTs)
    from engine import train_and_evaluate

    parser = argparse.ArgumentParser()
    get_args_parser(parser)
    out_dir = tempfile.mkdtemp()
    for gating in ['none', 'quantum']:
        run_dir = os.path.join(out_dir, gating)
        args = parser.parse_args(['--model', 'vit_tiny_patch16_224', '--epochs', '1', '--batch-size', '4', '--num_tasks', '2',
                                  '--prompt_gating', gating, '--output_dir', run_dir, '--save_ckpt', 'none', '--device', 'cpu'])
        args.distributed, args.nb_classes, args.print_freq = False, 4, 100
        os.makedirs(run_dir, exist_ok=True)
        torch.manual_seed(0)
        kwargs = dict(pretrained=False, num_classes=args.nb_classes)
        original_model = create_model(args.model, **kwargs)
        model = create_model(args.model, prompt_length=args.length, embedding_key=args.embedding_key, prompt_pool=True,
                             prompt_key=True, pool_size=args.size, top_k=args.top_k, batchwise_prompt=True,
                             head_type=args.head_type, prompt_gating=gating, **kwargs)
        for p in original_model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

        class_mask = [[0, 1], [2, 3]]
        loaders = []
        for classes in class_mask:
            y = torch.tensor(classes * 4)
            ds = torch.utils.data.TensorDataset(torch.rand(len(y), 3, 224, 224), y)
            loaders.append({'train': torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True),
                            'val': torch.utils.data.DataLoader(ds, batch_size=4)})
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        train_and_evaluate(model, model, original_model, torch.nn.CrossEntropyLoss(), loaders, optimizer, None,
                           torch.device('cpu'), class_mask, args)

        with open(os.path.join(run_dir, 'summary.json')) as f:
            summary = json.load(f)
        assert summary['completed_tasks'] == 2 and len(summary['epoch_times']) == 2
        assert summary['prompt_gating'] == gating
        expected_gate = 0 if gating == 'none' else args.size * (model.embed_dim + 1) + 2 * 2 * 3 * 3
        assert summary['n_gate_params'] == expected_gate, summary['n_gate_params']
        if gating == 'quantum':
            assert 'train_GateEnt' in summary['history'][-1] and 'test_GateDev' in summary['history'][-1]
        print(gating, {k: summary[k] for k in ['final_avg_acc', 'avg_incremental_acc', 'forgetting', 'n_gate_params']})


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    if '--fast' in sys.argv:
        tests = [t for t in tests if t is not test_train_and_evaluate_end_to_end]
    for test in tests:
        test()
        print('ok', test.__name__)
