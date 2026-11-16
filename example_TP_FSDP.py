import argparse
import logging
from dataclasses import asdict
from pathlib import Path
import json
import csv
import time
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
#分布式如何存储参数
import torch.distributed.checkpoint as dcp

from torch.distributed import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
parallelize_module,ColwiseParallel,RowwiseParallel
)
#这个的作用是什么
from torch.distributed.tensor import Replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
)
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.disttrain.model import MymodeConfig,DecoderLayer
from src.disttrain.data import TokenDataset

#参数工程化
def parse_args():
    p = argparse.ArgumentParser()


    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--output", default="outputs/tp2_fsdp2")
    p.add_argument("--resume", default=None)
    # 训练参数
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument('--max_seq_len', type=int, default=512)
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


    args = p.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    if not 1 <= args.seq_len <= 1024:
        p.error("seq-len must be in [1, 128]")
    if args.log_every < 1 or args.save_every < 0:
        p.error("invalid log-every or save-every")
    return args
#日志初始化
def init_log(log_dir:Path):
    loger=logging.getLogger()
    loger.setLevel(logging.DEBUG)
    formatter=logging.Formatter(
        fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    )
    file_handler=logging.FileHandler(
        log_dir / f'{time.strftime("%d-%H%M%S")}.log',
        encoding='utf-8',
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    loger.addHandler(file_handler)
    return loger
log_dir = Path('outputs/tp2_fsdp2/logs')
log_dir.mkdir(parents=True, exist_ok=True)
loger = init_log(log_dir)


#初始化通信组
def init_dist():
    #初始化通信组
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size=dist.get_world_size()
    rank=dist.get_rank()
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device=torch.device(local_rank)
    mesh=init_device_mesh('cuda',(2,2),mesh_dim_names=('DP','TP'))
    #每个rank分配到的信息是不同的
    #02 13
    dp_mesh=mesh["DP"]
    #01 23
    tp_mesh=mesh["TP"]
    #相对于本地组的位置
    dp_rank=dp_mesh.get_local_rank()
    tp_rank=tp_mesh.get_local_rank()
    if rank==2:
        #02,23,1,0
        loger.debug(f'dp_mesh{dp_mesh},tp_mesh{tp_mesh},dp_rank:{dp_rank},tp_rank:{tp_rank}')
    return dp_mesh,tp_mesh,dp_rank,tp_rank,device
#设计分布式策略
def apply_parallelism(model:DecoderLayer,cfg,tp_mesh:DeviceMesh,dp_mesh:DeviceMesh):
    tp_size=tp_mesh.size()
    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        #模型其他层要进行tp切分
        parallelize_plan={
            "emb": RowwiseParallel(
                input_layouts=Replicate(),
            ),
            "mlp": ColwiseParallel(
                output_layouts=Replicate(),
            ),
        },
    )
    #给模型每层配置
    for layer in model.layer:
        plan={
            "attention.q_proj":ColwiseParallel(),
            "attention.k_proj":ColwiseParallel(),
            "attention.v_proj":ColwiseParallel(),
            "attention.o_proj":RowwiseParallel(),
            "swiglu.gate_proj":ColwiseParallel(),
            "swiglu.up_proj":ColwiseParallel(),
            "swiglu.down_proj":RowwiseParallel(),
        }
        parallelize_module(layer,tp_mesh,parallelize_plan=plan)
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=True)
        layer.attention.n_heads = cfg.n_heads // tp_size
    #先对每层通信进行划分通信域，而后对整体模型其他参数进行划分
    fully_shard(model,mesh=dp_mesh,reshard_after_forward=False)
    #fsdp切分
    return model
import shutil
def save_checkpoint(model,optimizer,folder,step,args):
    folder = Path(folder)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dcp.save(
        state_dict={'model':model_sd, 'optimizer':optim_sd},
        checkpoint_id=str(folder),
    )
    if dist.get_rank() == 0:
        torch.save(
            {
                "step": step,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "seq_len": args.seq_len,
                "lr": args.lr,
                "bf16": args.bf16,
            },
            folder / "trainer.pt",
        )
    dist.barrier()
def load_checkpoint(model,optimizer,folder,args):
    folder = Path(folder)
    meta=torch.load(
        folder / "trainer.pt",
        map_location="cpu",

    )
    for key in ("batch_size", "seed", "seq_len", "lr", "bf16"):
        if meta[key] != getattr(args, key):
            raise ValueError(f"Resume configuration mismatch: {key}")
    model_sd, optim_sd = get_state_dict(model, optimizer)
    state = {"model": model_sd, "optimizer": optim_sd}

    # DCP 将分片数据加载回当前设备网格
    dcp.load(state_dict=state, checkpoint_id=str(folder))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    return meta["step"]

def main():
    args = parse_args()
    if torch.cuda.device_count() < 4:
        raise RuntimeError("This experiment requires four CUDA GPUs")
    dp_mesh,tp_mesh,dp_rank,tp_rank,device=init_dist()
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        #创建模型，优化器
        model_cfg = MymodeConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.max_seq_len,
            mlp_hidden_dim=args.mlp_hidden_dim,
        )
        model=DecoderLayer(model_cfg).to(device)
        model=apply_parallelism(model,model_cfg,tp_mesh,dp_mesh)
        #分配结束后
        # 64 64
        if dist.get_rank() == 0:
            loger.debug(f'model.layer.0.attention.q_proj{model.layer[0].attention.q_proj.weight.to_local().shape}')
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_step = 0
        if args.resume:
            start_step = load_checkpoint(
                model, optimizer, args.resume, args
            )
        global_batch = args.batch_size * dp_mesh.size()
        total_samples = args.steps * global_batch
        #数据元信息
        # 写入运行时候元信息
        run_signature = {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "world_size": dist.get_world_size(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
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
        if dist.get_rank() == 0:
            # 运行时 元信息
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_config.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        #创建数据集
        dataset=TokenDataset(total_samples,args.seq_len,args.vocab_size,seed=args.seed)
        remaining = Subset(
            dataset,
            range(start_step * global_batch, total_samples),
        )
        sampler=DistributedSampler(remaining,num_replicas=dp_mesh.size(),rank=dp_rank,shuffle=False)
        loader=DataLoader(
            remaining,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        out_dir=Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        fields = [
            "step", "loss",
            "step_ms", "tokens_per_s", "peak_alloc_mb",
        ]
        log_file = None
        if dist.get_rank() == 0:
            log_path = out_dir / "metrics.csv"
            append = bool(args.resume) and log_path.exists()
            log_file = log_path.open(
                "a" if append else "w",
                newline="",
                encoding="utf-8",
            )
            writer = csv.DictWriter(log_file, fieldnames=fields)
            if not append:
                writer.writeheader()
        try:
            loader_iter = iter(loader)
            for step in range(start_step, args.steps):
                t0=time.perf_counter()
                x,y=next(loader_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.bf16,
                ):
                    logits=model(x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, model_cfg.vocab_size).float(),
                        y.reshape(-1),
                    )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t0

                should_log = (
                        (step + 1) % args.log_every == 0
                        or step == start_step
                        or step + 1 == args.steps
                )
                if should_log:
                    mean_loss=loss.detach().float().clone()
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss /= dist.get_world_size()
                    max_time = torch.tensor(elapsed, device=device)
                    dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
                    peak_mb = torch.tensor(
                        torch.cuda.max_memory_allocated(device) / 2 ** 20,
                        device=device,
                    )
                    dist.all_reduce(peak_mb, op=dist.ReduceOp.MAX)

                    tokens_s = (
                            global_batch * args.seq_len / max_time.item()
                    )

                    if dist.get_rank() == 0:
                        record = {
                            "step": step + 1,
                            "loss": mean_loss.item(),
                            "step_ms": max_time.item() * 1000,
                            "tokens_per_s": tokens_s,
                            "peak_alloc_mb": peak_mb.item(),
                        }
                        writer.writerow(record)
                        log_file.flush()
                        print(record, flush=True)
                if args.save_every > 0 and (
                        (step + 1) % args.save_every == 0
                        or step + 1 == args.steps
                ):
                    path = out_dir / f"ckpt_step_{step + 1:06d}"
                    save_checkpoint(
                        model, optimizer, path, step + 1, args)
        finally:
            if log_file is not None:
                log_file.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()