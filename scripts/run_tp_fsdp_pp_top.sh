set -euo pipefail
cd "$(dirname "$0")/.."


torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  example_dp_pp_tp.py \
  --steps 100 \
  --batch-size 256 \
  --seq-len 1024 \
  --n-layers 16\
  --n-heads 32\
  --output outputs/dp_pp_tp_top