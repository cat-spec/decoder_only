import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self,num_samples:int,seq_len:int,vocab_size:int,seed:int=42):
        if num_samples <= 0 or seq_len <= 0:
            raise ValueError("num_samples/seq_len must be > 0")
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        #数据集大小 steps*16
        self.num_samples = num_samples
        #token的长度
        self.seq_len = seq_len
        #词汇表大小
        self.vocab_size = vocab_size
        self.seed = seed
        self.positions=torch.arange(seq_len+1,dtype=torch.long)
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)

        # 只由 index 和 seed 决定样本
        start = (index * 37 + self.seed) % self.vocab_size

        tokens = (start + self.positions
                 ) % self.vocab_size

        x = tokens[:-1].clone()
        y = tokens[1:].clone()

        return x, y