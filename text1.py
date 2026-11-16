import torch
from torch.distributed.tensor import Replicate
from torch.distributed.fsdp import fully_shard

print(torch.__version__)
print(Replicate)