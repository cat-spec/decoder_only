import argparse
import logging
import time
from pathlib import Path
import os
import json
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import Subset, DataLoader, DistributedSampler

import torch.distributed as dist
import torch.distributed.checkpoint.state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (parallelize_module,ColwiseParallel,RowwiseParallel)
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage,Schedule1F1B
from torch.utils.tensorboard import writer

from src.disttrain.model import MymodeConfig,DecoderLayer
from src.disttrain.data import TokenDataset
from src.disttrain.model.decoder_only import TransformerBlock,RMSNorm,RotaryEmbedding

class BlockStage(nn.Module):
    def __init__(self,cfg:MymodeConfig,pp_mesh:DeviceMesh,pp_rank,seed):
        super().__init__()
        self.rope=RotaryEmbedding(cfg.d_model//cfg.n_heads,cfg.rope_theta)
        self.blocks=nn.ModuleDict()
        block_per_stage=cfg.n_layers//pp_mesh.size()
        start=pp_rank*block_per_stage
        end=start+block_per_stage
        for i in range(start,end):
            torch.manual_seed(seed+100+i)
            #生成一个层
            block=TransformerBlock(cfg)
            block.apply(DecoderLayer._init_weights)
            self.blocks[str(i)] = block
    def forward(self,hidden):
        logger=logging.getLogger(__name__)
        logger.debug(f"hidden: {hidden.shape}")
        cos, sin = self.rope(hidden.shape[1], hidden.dtype)

        for block in self.blocks.values():
            hidden = block(hidden, cos, sin)
        return hidden
class StageAdapter(nn.Module):
    def __init__(self,blocks,boundary,pp_rank):
        super().__init__()
        self.rank=pp_rank
        self.blocks=blocks
        self.boundary=boundary
    def forward(self,x):
        #初始流水线
        if self.rank==0:
            #前向emb
            x=self.boundary(x)
        x=self.blocks(x)
        if self.rank==1:
            x=self.boundary(x)
        return x
class OutputHead(nn.Module):
    def __init__(self,cfg:MymodeConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
        )
        self.lm_head.apply(DecoderLayer._init_weights)
    def forward(self,hidden):
        return self.lm_head(self.norm(hidden))
# 参数工程化
def parse_args():
    p=argparse.ArgumentParser()
    # 训练参数
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument('--max_seq_len',type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    # p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # 模型参数
    p.add_argument("--vocab-size", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mlp-hidden-dim", type=int, default=1024)
    p.add_argument('--microbatches',type=int,default=4)
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
    p.add_argument("--output", default="outputs/dp_pp_tp")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--resume", type=Path, default=None)

    return p.parse_args()
#初始化日志
def init_logger(log_dir:Path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(str(log_dir / f"log.txt{dist.get_rank()}.log"))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    return logger
# 初始化通信组
def init_dist():
    logger=logging.getLogger(__name__)
    mesh=init_device_mesh('cuda',(2,2,2),mesh_dim_names=('DP','PP','TP'))
    dp_mesh=mesh['DP']
    pp_mesh=mesh['PP']
    tp_mesh=mesh['TP']
    dp_rank=dp_mesh.get_local_rank()
    pp_rank=pp_mesh.get_local_rank()
    tp_rank=tp_mesh.get_local_rank()
    #rank0输出到日志
    if dist.get_rank()==2:
        #26 02 23
        logger.debug(f"dp_mesh: {dp_mesh}, pp_mesh: {pp_mesh}, tp_mesh: {tp_mesh}")
        #0 1 0
        logger.debug(f"dp_rank: {dp_rank}, pp_rank: {pp_rank}, tp_rank: {tp_rank}")
    return dp_mesh,pp_mesh,tp_mesh,dp_rank,pp_rank,tp_rank
def init_csv(output:Path,args):
    metrics_path = output / "metrics.csv"
    append = bool(args.resume) and metrics_path.exists()
    log_file = metrics_path.open(
        "a" if append else 'w', newline="", encoding="utf-8"
    )
    writer = csv.DictWriter(log_file, fieldnames={'step','step_ms','tokens/s','max_mem'})
    if not append:
        writer.writeheader()
    return writer,log_file
# 分布式参数
def apply_parallelism(model):
    """
    stage 0:block 0 1
    stage 1:block 2 3
    """
def main():
    args = parse_args()
    #路径初始化
    output = Path(args.output)
    checkpoint_dir=output/'checkpoint'
    log_dir=output/'log'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    #通信组初始化
    dist.init_process_group(backend='nccl')
    world_size=dist.get_world_size()
    rank=dist.get_rank()
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device=torch.device(local_rank)
    dp_mesh,pp_mesh,tp_mesh,dp_rank,pp_rank,tp_rank=init_dist()

    #日志初始化
    logger=init_logger(log_dir)
    logger.debug(f"pp_mesh: {pp_mesh.size()}")
    #模型参数初始化
    cfg=MymodeConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        mlp_hidden_dim=args.mlp_hidden_dim,
    )
    #初始化流水线模型
    #0,1,4获得01层数 2获得23层数
    stage=BlockStage(cfg,pp_mesh,pp_rank,args.seed).to(device)
    #流水线0需要emb
    if pp_rank==0:
        torch.manual_seed(4242)
        boundary=nn.Embedding(cfg.vocab_size,cfg.d_model).to(device)
        boundary.apply(DecoderLayer._init_weights)
    #其他流水线负责输出
    else:
        torch.manual_seed(4243)
        boundary = OutputHead(cfg).to(device)
    for block in stage.blocks.values():
        #tp切分
        plan = {
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.o_proj": RowwiseParallel(),
            "swiglu.gate_proj": ColwiseParallel(),
            "swiglu.up_proj": ColwiseParallel(),
            "swiglu.down_proj": RowwiseParallel(),
        }
        parallelize_module(block,device_mesh=tp_mesh,parallelize_plan=plan)
        block.attention.n_heads = (
                cfg.n_heads // tp_mesh.size()
        )
        #fsdp切分
        fully_shard(block,mesh=dp_mesh,reshard_after_forward=True)
    fully_shard(boundary,mesh=dp_mesh,reshard_after_forward=False)
    #pp切分
    execution_model=StageAdapter(stage,boundary, pp_rank)
    pipeline_stage=PipelineStage(
        submodule=execution_model,
        stage_index=pp_rank,
        num_stages=pp_mesh.size(),
        device=device,
        group=pp_mesh.get_group()
    )
    # 优化器管理当前进程真正拥有的模块
    optimizer = torch.optim.AdamW(
        execution_model.parameters(),
        lr=args.lr,
    )
    #初始化调度器
    schedule = Schedule1F1B(
        pipeline_stage,
        n_microbatches=args.microbatches,
        loss_fn=lambda logits, targets: F.cross_entropy(
            logits.float().reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
        ),
    )
    #初始化数据
    global_batch = args.batch_size * dp_mesh.size()
    dataset = TokenDataset(
        num_samples=args.steps * global_batch,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
    )
    sampler=DistributedSampler(dataset,num_replicas=dp_mesh.size(),rank=dp_rank,shuffle=False,seed=args.seed)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,shuffle=False,num_workers=4,pin_memory=True)
    micro_size=args.batch_size//args.microbatches
    # PP 子网格中另一个 Rank 的全局编号
    stage.train()
    boundary.train()
    if rank==0:
        writer,csv_file = init_csv(output, args)
    torch.cuda.reset_max_memory_allocated(device=device)
    for step,(ids,lables) in enumerate(dataloader):
        ids,lables=ids.to(device,non_blocking=True),lables.to(device,non_blocking=True)
        optimizer.zero_grad()
        t0=time.perf_counter()
        losses = []
        if pp_rank==0:
            schedule.step(ids)
        else:
            schedule.step(target=lables,losses=losses)
            logger.debug(f'losses: {losses}')
        # 所有 Microbatch 完成后只更新一次
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            time.perf_counter() - t0,
            device=device,
        )
        mem=torch.tensor(torch.cuda.max_memory_allocated(device=device)/1024**2,device=device)
        dist.all_reduce(mem,op=dist.ReduceOp.MAX)
        # 一个 Step 以最慢 Rank 为准
        dist.all_reduce(
            elapsed,
            op=dist.ReduceOp.MAX,
        )
        # 只有 PP Stage 1 计算 loss
        # 两个 TP Rank 上的 loss 是重复的
        if rank == 0:
            tokens_s = (
                global_batch * args.seq_len / elapsed.item()
            )
            dist_csv={
                'step': step,
                'step_ms': elapsed.item(),
                'tokens/s': tokens_s,
                'max_mem':mem.item()
            }
            writer.writerow(dist_csv)
            csv_file.flush()
            print(
                f"step={step+1:03d} "
                f"step_ms={elapsed.item()*1000:.2f} "
                f"tokens/s={tokens_s:.1f}",
                f'mem={mem.item():.2f} MB',
                flush=True,
            )
if __name__ == "__main__":
    main()
    dist.destroy_process_group()





