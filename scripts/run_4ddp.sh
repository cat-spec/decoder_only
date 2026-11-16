
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# 只暴露两张指定的 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 分布式启动
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  example_2DDP.py \
  --steps "${STEPS:-200}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --seq-len 512 \
  --n-layers 4\
  --n-heads 8\
  --amp "${AMP:-none}" \
  --lr 3e-4 \
  --weight-decay 0.1 \
  --grad-clip 1.0 \
  --seed 42 \
  --log-every 10 \
  --output outputs/ddp4
