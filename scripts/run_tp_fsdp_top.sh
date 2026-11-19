set -euo pipefail
cd "$(dirname "$0")/.."


torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  example_TP_FSDP.py \
  --steps 100 \
  --batch-size 128 \
  --seq-len 512 \
  --n-layers 4\
  --n-heads 8\
  --output outputs/tp2_fsdp2_top_2080