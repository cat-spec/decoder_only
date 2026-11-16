import argparse
import csv
import json
import time
from pathlib import Path
from dataclasses import asdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset

from src.disttrain.data import TokenDataset
from src.disttrain.model import MymodeConfig,DecoderLayer

FIELDS = [
    "step", "loss", "lr", "grad_norm",
    "data_ms", "h2d_ms", "compute_ms",
    "tokens_per_s", "end_to_end_tokens_per_s",
    "peak_alloc_mb", "peak_reserved_mb",
]

def parse_args():
    p=argparse.ArgumentParser()

    # 训练参数
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument('--max_seq_len',type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # 模型参数
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)

    # 环境和混合精度
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--amp",
        choices=["none", "bf16", "fp16"],
        default="none",
    )
    p.add_argument("--num-workers", type=int, default=0)

    # 工程管理
    p.add_argument("--output", default="outputs/tmp")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume", type=Path, default=None)

    return p.parse_args()

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
#保留模型参数
def save_checkpoint(
    path, step, model, optimizer,
    scheduler, scaler, model_cfg,
    run_signature, device,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(model_cfg),
        "run_signature": run_signature,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda" else None
        ),
    }

    # 先写临时文件，再替换目标文件
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)

def load_checkpoint(path, model, optimizer,
    scheduler, scaler, model_cfg,
    run_signature, device,):
    state=torch.load(path, map_location="cpu",weights_only=True)
    if state["model_config"] != asdict(model_cfg):
        raise ValueError("Checkpoint model config mismatch")

    if state["run_signature"] != run_signature:
        raise ValueError("Checkpoint training config mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("Checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state(
            state["cuda_rng"], device
        )
    #续训练轮数
    return int(state["step"])

def prepare_metrics(path, resume_step):
    # 恢复旧 checkpoint 时，删除其后可能存在的旧日志
    previous_rows = []
    if resume_step > 0 and path.exists():
        with path.open("r", newline="") as f:
            previous_rows = [
                row for row in csv.DictReader(f)
                if int(row["step"]) <= resume_step
            ]

    f = path.open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(previous_rows)
    f.flush()
    return f, writer

def main():
    args = parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if args.seq_len <= 0 or args.log_every <= 0:
        raise ValueError("seq_len/log_every must be positive")
    if args.save_every <= 0 or args.num_workers < 0:
        raise ValueError("invalid save_every/num_workers")
    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("lr/grad_clip must be positive")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP requires CUDA in this demo")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    # 随机种子
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    # 2. 输出目录
    output = Path(args.output)  # outputs/single
    # 确保文件夹存在
    output.mkdir(parents=True, exist_ok=True)
    ##outputs/single/checkpoints
    ckpt_dir = output / "checkpoints"

    model_cfg = MymodeConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )

    model = DecoderLayer(model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr * 0.1,
    )

    # 5. 混合精度
    use_amp = args.amp != "none"
    amp_dtype = (
        torch.bfloat16
        if args.amp == "bf16"
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(args.amp == "fp16")
    )

    run_signature = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "amp": args.amp,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    }

    metadata = {
        "model": asdict(model_cfg),
        "training": run_signature,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else None
        ),
    }
    # 运行时 元信息
    (output / "run_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # 6. 恢复训练状态
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume, model, optimizer,
            scheduler, scaler, model_cfg,
            run_signature, device,
        )
        print(f"Resumed from step {start_step}")

    if start_step >= args.steps:
        print("Training already completed")
        return

    # 7. 构建确定性数据集
    # 总轮数*批次大小
    total_samples = args.steps * args.batch_size
    # (x,y)
    dataset = TokenDataset(
        num_samples=total_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    # 从上次完成的 step 对应位置继续
    indices = range(
        start_step * args.batch_size,
        total_samples,
    )
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        generator=torch.Generator().manual_seed(args.seed),
    )

    iterator = iter(loader)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Device: {device} | AMP: {args.amp}")

    # 8. 日志文件
    csv_file, writer = prepare_metrics(
        output / "metrics.csv", start_step
    )

    try:
        for step in range(start_step + 1, args.steps + 1):
            t0 = time.perf_counter()
            x, y = next(iterator)
            data_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            x = x.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            y = y.to(
                device,
                non_blocking=(device.type == "cuda"),
            )
            # print(x.shape, y.shape)
            synchronize(device)
            h2d_ms = (time.perf_counter() - t0) * 1000
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            # Forward + Loss
            with amp_context:
                # (b,s,v)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, args.vocab_size),
                    y.reshape(-1),
                )
            # Backward
            scaler.scale(loss).backward()
            # 在梯度裁剪前先恢复梯度量级
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
                error_if_nonfinite=(args.amp != "fp16"),
            )
            # 记录本次真正使用的 lr
            lr_used = optimizer.param_groups[0]["lr"]
            # Optimizer Step
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # FP16 溢出时可能跳过 optimizer step
            did_step = (
                    args.amp != "fp16"
                    or scaler.get_scale() >= old_scale
            )
            if did_step:
                scheduler.step()
            synchronize(device)
            compute_ms = (time.perf_counter() - t0) * 1000
            # 性能指标
            loss_value = float(loss.detach().item())
            grad_norm_value = float(grad_norm.item())
            # 吞吐量
            num_tokens = x.numel()
            tokens_per_s = (
                    num_tokens / (compute_ms / 1000)
            )
            end_to_end_ms = data_ms + h2d_ms + compute_ms
            end_to_end_tps = (
                    num_tokens / (end_to_end_ms / 1000)
            )
            # 内存
            if device.type == "cuda":
                peak_alloc = (
                        torch.cuda.max_memory_allocated(device)
                        / 1024 ** 2
                )
                peak_reserved = (
                        torch.cuda.max_memory_reserved(device)
                        / 1024 ** 2
                )
            else:
                peak_alloc = 0.0
                peak_reserved = 0.0
            writer.writerow({
                "step": step,
                "loss": loss_value,
                "lr": lr_used,
                "grad_norm": grad_norm_value,
                "data_ms": data_ms,
                "h2d_ms": h2d_ms,
                "compute_ms": compute_ms,
                "tokens_per_s": tokens_per_s,
                "end_to_end_tokens_per_s": end_to_end_tps,
                "peak_alloc_mb": peak_alloc,
                "peak_reserved_mb": peak_reserved,
            })
            csv_file.flush()
            if step == 1 or step % args.log_every == 0:
                print(
                    f"step={step:04d} "
                    f"loss={loss_value:.4f} "
                    f"lr={lr_used:.2e} "
                    f"grad={grad_norm_value:.3f} "
                    f"compute={compute_ms:.2f}ms "
                    f"tok/s={tokens_per_s:.0f} "
                    f"peak={peak_alloc:.1f}MiB"
                )
        # E. Checkpoint
        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = (
                    ckpt_dir / f"step_{step:06d}.pt"
            )
            save_checkpoint(
                ckpt_path, step, model, optimizer,
                scheduler, scaler, model_cfg,
                run_signature, device,
            )
            print(f"Saved: {ckpt_path}")

    finally:
        csv_file.close()

"""
python example_singletrain.py \
    --device cuda \
    --amp bf16 \
    --steps 1000 \
    --batch-size 8 \
    --seq-len 64 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --seed 42 \
    --num-workers 0 \
    --log-every 100 \
    --save-every 200 \
    --output outputs/single
"""



if __name__ == "__main__":
    main()
