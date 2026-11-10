import torch
import torch.nn as nn
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MymodeConfig

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        # self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x32=x.float()
        normmalized=x32*torch.rsqrt(x32.square().mean(dim=-1,keepdim=True)+self.eps)
        return normmalized.to(x.dtype)*self.weight.to(x.dtype)

class RotaryEmbedding(nn.Module):
    """Interleaved RoPE: feature pairs (0,1), (2,3), ... are rotated."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        angles = torch.outer(positions, self.inv_freq)
        return angles.cos().to(dtype), angles.sin().to(dtype)
def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: [B,H,T,D]; cos/sin: [T,D/2]."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

class Causal_Attention(nn.Module):
    def __init__(self, cfg:MymodeConfig):
        super().__init__()
        #头数设置为4
        self.n_heads = cfg.n_heads
        #每头维度为256//4 =64
        self.head_dim = cfg.d_model // cfg.n_heads
        #每个qkv是256*256
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    def forward(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
        batch_size,seq_len=x.shape[:2]
        #把x(B,S,D)转为思维(B,S,H,head_dim)后变形为(B,H,S,head_dim)
        def split_heads(linear:nn.Linear) -> Tensor:
            return linear(x).reshape(batch_size,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        q, k, v = split_heads(self.q_proj), split_heads(self.k_proj), split_heads(self.v_proj)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        #输入(B,H,S,D)维度
        attn_out=F.scaled_dot_product_attention(q, k, v,dropout_p=0.0,is_causal=True)
        #输出维度信息不变，转变为(B,S,H,D)在合并头维度
        attn_out=attn_out.transpose(1,2).contiguous().reshape(batch_size,seq_len,-1)

        return self.o_proj(attn_out)
class SwiGlUMLP(nn.Module):
    def __init__(self, cfg:MymodeConfig):
        super().__init__()
        self.gate_proj=nn.Linear(cfg.d_model,cfg.mlp_hidden_dim,bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.mlp_hidden_dim, bias=False)
        self.down_proj=nn.Linear(cfg.mlp_hidden_dim, cfg.d_model, bias=False)
    def forward(self, x:Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
class TransformerBlock(nn.Module):
    def __init__(self, cfg:MymodeConfig):
        super().__init__()
        self.rmsnorm=RMSNorm(cfg.d_model)
        self.attention=Causal_Attention(cfg)
        self.swiglu=SwiGlUMLP(cfg)
    def forward(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
        x=x+self.attention(self.rmsnorm(x),cos,sin)
        x=x+self.swiglu(self.rmsnorm(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, cfg:MymodeConfig):
        super().__init__()
        self.cfg=cfg
        self.layer=nn.ModuleList(TransformerBlock(cfg)for _ in range(cfg.n_layers))
        self.emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.final_norm=RMSNorm(cfg.d_model)
        self.mlp=nn.Linear(cfg.d_model, cfg.vocab_size,bias=False)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, x:Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        seq_len=x.shape[1]
        x=self.emb(x)
        cos, sin = self.rope(seq_len,x.dtype)
        for layer in self.layer:
            x=layer(x,cos,sin)
        return self.mlp((self.final_norm(x)))
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

