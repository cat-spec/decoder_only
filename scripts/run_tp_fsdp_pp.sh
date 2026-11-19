set -euo pipefail
cd "$(dirname "$0")/.."


torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  example_dp_pp_tp.py \
  --steps 100 \
  --batch-size 128 \
  --seq-len 512 \
  --n-layers 4\
  --n-heads 8\
  --output outputs/dp_pp_tp