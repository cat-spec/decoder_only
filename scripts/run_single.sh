
#!/usr/bin/env bash
set -euo pipefail

# 确保从项目根目录运行
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-none}"

python example_singletrain.py \
    --device "$DEVICE" \
    --amp "$AMP" \
    --steps 200 \
    --batch-size 128 \
    --seq-len 512 \
    --n-layers 4\
    --n-heads 8\
    --lr 3e-4 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --seed 42 \
    --num-workers 0 \
    --log-every 10 \
    --save-every 20 \
    --output outputs/single
