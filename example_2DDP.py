import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler,DataLoader,Subset

import argparse
import csv
import json
import time
from pathlib import Path
from dataclasses import asdict
from contextlib import nullcontext

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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument('--max_seq_len',type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # 模型参数
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)

    # 环境和混合精度
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--amp",
        choices=["none", "bf16", "fp16"],
        default="none",
    )
    p.add_argument("--num-workers", type=int, default=0)

    # 工程管理
    p.add_argument("--output", default="outputs/ddp")
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
def check_parameters(ddp, device):
    # 训练结束后逐个参数与 Rank 0 对照
    max_diff = torch.zeros((), device=device)
    with torch.no_grad():
        for p in ddp.module.parameters():
            reference = p.detach().clone()
            dist.broadcast(reference, src=0)
            max_diff = torch.maximum(
                max_diff, (p.detach() - reference).abs().max()
            )
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    return max_diff.item()
def main():
    args = parse_args()
    if not args.cpu and torch.cuda.device_count() < 2:
        raise RuntimeError("Two visible CUDA GPUs required; use --cpu for Gloo smoke test")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if args.seq_len <= 0 or args.log_every <= 0:
        raise ValueError("seq_len/log_every must be positive")
    if args.save_every <= 0 or args.num_workers < 0:
        raise ValueError("invalid save_every/num_workers")
    if args.lr <= 0 or args.grad_clip <= 0:
        raise ValueError("lr/grad_clip must be positive")

    # 创建通信组
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("AMP requires CUDA in this demo")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    # 随机种子 设立同一个种子
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    model_cfg = MymodeConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )

    model = DecoderLayer(model_cfg).to(device)
    #对模型进行包裹通信
    model=DDP(model,device_ids=[local_rank])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda _: 1.0
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
    #写入运行时候元信息
    run_signature = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "world_size": world_size,
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
    if rank==0:
        # 运行时 元信息
        output=Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    # 6. 恢复训练状态
    start_step = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=True
        )
        if checkpoint["signature"] != run_signature:
            raise ValueError("Checkpoint configuration mismatch")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = checkpoint["step"]
    # 7. 构建确定性数据集
    # 总轮数*批次大小
    global_batch = args.batch_size * world_size
    total_samples = args.steps * global_batch
    # (x,y)
    dataset = TokenDataset(
        num_samples=total_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    # 从上次完成的 step 对应位置继续
    remaining = Subset(
        dataset,
        range(start_step * global_batch, total_samples),
    )
    #for(rank::world_size)进行分发数据
    sampler=DistributedSampler(remaining,num_replicas=world_size,rank=rank,shuffle=False,drop_last=True)
    loader = DataLoader(
        remaining,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
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
    out_dir = Path(args.output)
    fields = [
        "step", "loss", "lr", "grad_norm",
        "step_ms", "tokens_per_s", "peak_alloc_mb"
    ]
    log_file=None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.csv"
        append = bool(args.resume) and metrics_path.exists()
        log_file = metrics_path.open(
            "a" if append else "w", newline="", encoding="utf-8"
        )
        writer = csv.DictWriter(log_file, fieldnames=fields)
        if not append:
            writer.writeheader()
        print(f"world_size={world_size}, global_batch={global_batch}")
    #开启训练
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
            forward_time = (time.perf_counter() - t0) *1000
            # Backward
            t0 = time.perf_counter()
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
            backword_time = (time.perf_counter() - t0) * 1000


            # 吞吐量

            end_to_end_ms = data_ms + h2d_ms + forward_time+backword_time

            should_log = (
                    (step ) % args.log_every == 0
                    or step == start_step
                    or step + 1 == args.steps
            )
            if should_log:
                mean_loss = loss.detach().float().clone()
                dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                mean_loss /= world_size

                max_time = torch.tensor(end_to_end_ms, device=device)
                dist.all_reduce(max_time, op=dist.ReduceOp.MAX)

                memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.type == "cuda" else 0.0
                )
                peak_mem = torch.tensor(memory_mb, device=device)
                dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

                tokens_per_s = global_batch * args.seq_len*1000 / max_time.item()
                if rank==0:
                    record = {
                        "step": step,
                        "loss": mean_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm.item(),
                        "step_ms": max_time.item(),
                        "tokens_per_s": tokens_per_s,
                        "peak_alloc_mb": peak_mem.item(),
                    }
                    writer.writerow(record)
                    log_file.flush()
                    print(
                        f"step={step:04d} "
                        f"loss={mean_loss:.4f} "
                        f"lr={lr_used:.2e} "
                        f"grad={grad_norm:.3f} "
                        f"forward_time={forward_time:.2f}ms "
                        f'backword_time={backword_time:.2f}ms '
                        f"tok/s={tokens_per_s:.0f} "
                        f"peak={peak_mem:.1f}MiB"
                    )
        # E. Checkpoint
        max_diff = check_parameters(model, device)
        if rank == 0:
            print(f"Max parameter difference between ranks: {max_diff:.3e}")
            state = {
                "step": args.steps,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "signature": run_signature,
            }
            tmp_path = out_dir / "last.pt.tmp"
            torch.save(state, tmp_path)
            os.replace(tmp_path, out_dir / "last.pt")

        dist.barrier()

    finally:
        if log_file is not None:
            log_file.close()
        dist.destroy_process_group()

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
