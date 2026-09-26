#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; extra CLI arguments at the end override this preset.
cd "$(dirname "$0")"
ROUTER="${ROUTER:-qsd_comp}"
SEED="${SEED:-10961}"
EPOCHS="${EPOCHS:-7}"
DATA_PATH="${DATA_PATH:-/kaggle/working/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/output_${ROUTER}_e${EPOCHS}_seed${SEED}}"

python main.py cifar100_l2p \
    --prompt_router "$ROUTER" \
    --seed "$SEED" --epochs "$EPOCHS" --batch-size 16 \
    --data-path "$DATA_PATH" --output_dir "$OUTPUT_DIR" \
    --no_batchwise_prompt --length 5 --top_k 5 \
    --qsd_state_dim 32 --qsd_rank 4 --qsd_eps 0.001 \
    --qsd_cls_mix 0.5 --qsd_cosine_tau 0.1 \
    --comp_components_per_task 5 --comp_quantum_mix 0.5 \
    --comp_prototypes_per_class 4 --comp_candidates_per_class 64 \
    --comp_memory_batch_size 32 \
    --comp_retention_coeff 1.0 --comp_prompt_coeff 1.0 \
    "$@"
