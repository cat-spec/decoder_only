
set -euo pipefail
cd "$(dirname "$0")/.."


#export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  example_TP_FSDP.py \
  --steps 100 \
  --batch-size 128 \
  --seq-len 512 \
  --n-layers 4\
  --n-heads 8\
  --output outputs/tp2_fsdp2

