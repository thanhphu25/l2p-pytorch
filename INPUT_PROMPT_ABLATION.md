# Input-space prompt ablation

This branch adds an input-space prompt before ViT patch embedding while leaving
the original L2P token-prompt path intact. It is inspired by LGSP rather than a
line-for-line reproduction. Reference:
[paper](https://arxiv.org/abs/2507.09183) and
[official code](https://github.com/Jywsuperman/LGSP).

- A compact convolutional generator creates a pool of local spatial prompts on
  the 14 x 14 patch grid and upsamples the selected mixture to the image.
- An optional global branch learns weights for concentric Fourier rings. Its
  uniform initialization is an all-pass filter, so its initial residual is zero.
- The frozen ViT CLS feature used by L2P also supplies the routing query.
- The quantum-inspired router amplitude-encodes top-k cosine scores, phase-encodes
  the query, applies trainable RY/RZ and controlled-phase operations, and uses
  simulated measurement probabilities as mixture weights.

This is a differentiable state-vector simulation on ordinary GPU hardware. It
does not use a quantum computer and should be described as quantum-inspired
unless experiments are later run on quantum hardware.

## Controlled experiments

Use the same seed and all original L2P hyperparameters:

| Experiment | Local prompt | Fourier prompt | Router |
| --- | --- | --- | --- |
| baseline | no | no | original L2P only |
| local | yes | no | cosine |
| local_global | yes | yes | cosine |
| linear | yes | yes | learned linear |
| quantum_no_phase | yes | yes | circuit, no query phase |
| quantum | yes | yes | full circuit |

Run one experiment on Kaggle:

~~~bash
chmod +x run_input_prompt_ablation.sh
DATA_PATH=/kaggle/input/your-cifar100-path \
  bash run_input_prompt_ablation.sh quantum
~~~

Extra CLI arguments are forwarded, for example:

~~~bash
DATA_PATH=/kaggle/input/your-cifar100-path SEED=10961 \
  bash run_input_prompt_ablation.sh local_global --batch-size 32
~~~

Recommended order is: `baseline`, `local`, `local_global`, `linear`,
`quantum_no_phase`, then `quantum`. The comparisons answer different
questions:

1. `local - baseline`: does input-space prompting help at all?
2. `local_global - local`: does the LGSP-style frequency branch help?
3. `linear - local_global`: does extra learned routing capacity explain gains?
4. `quantum - linear`: does interference routing beat a parameter-matched
   classical router?
5. `quantum - quantum_no_phase`: does query-dependent phase encoding matter?

Use at least three seeds before making a paper claim. Report final average
accuracy, average incremental accuracy, forgetting, trainable parameters, and
wall-clock time. A quantum contribution is supported only if `quantum`
consistently improves over both `linear` and `quantum_no_phase`, not merely
over the original baseline.
