#!/usr/bin/env bash
set -euo pipefail

# Run one controlled experiment at a time:
#   DATA_PATH=/kaggle/input/cifar100-python bash run_input_prompt_ablation.sh quantum
# Any arguments after the experiment name are forwarded to main.py.

experiment="${1:-}"
if [[ -z "${experiment}" ]]; then
    echo "usage: DATA_PATH=/path/to/data bash $0 {baseline|local|local_global|linear|quantum_no_phase|quantum} [extra args]"
    exit 2
fi
shift

data_path="${DATA_PATH:-./data}"
output_root="${OUTPUT_ROOT:-./output/input_prompt_ablation}"
seed="${SEED:-10961}"
mode="none"
global_args=()

case "${experiment}" in
    baseline)
        mode="none"
        ;;
    local)
        mode="cosine"
        ;;
    local_global)
        mode="cosine"
        global_args+=(--input-prompt-global)
        ;;
    linear)
        mode="linear"
        global_args+=(--input-prompt-global)
        ;;
    quantum_no_phase)
        mode="quantum_no_phase"
        global_args+=(--input-prompt-global)
        ;;
    quantum)
        mode="quantum"
        global_args+=(--input-prompt-global)
        ;;
    *)
        echo "unknown experiment: ${experiment}"
        exit 2
        ;;
esac

python main.py cifar100_l2p \
    --model vit_base_patch16_224 \
    --batch-size 16 \
    --data-path "${data_path}" \
    --seed "${seed}" \
    --output_dir "${output_root}/${experiment}/seed_${seed}" \
    --input-prompt-mode "${mode}" \
    "${global_args[@]}" \
    "$@"
