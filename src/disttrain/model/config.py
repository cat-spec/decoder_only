from dataclasses import dataclass

#模型基础配置
@dataclass(frozen=True)
class MymodeConfig:
    #词表大小
    vocab_size: int = 1024
    #词向量维度
    d_model: int = 256
    #层数大小
    n_layers: int = 8
    #头数
    n_heads: int = 4
    #模型能接受的最大seq长度
    max_seq_len: int = 128
    #隐藏层维度
    mlp_hidden_dim: int = 1024
    #归一化，防止分母为0
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    #dataclass自动初始化以上数据调用init后自动调用__post_init__
    def __post_init__(self):
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if min(self.d_model, self.n_layers, self.n_heads, self.max_seq_len, self.mlp_hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rms_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_eps and rope_theta must be positive")
