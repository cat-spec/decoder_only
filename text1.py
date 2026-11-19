import torch
a=torch.randn(2,3)
x=torch.stack([a[i] for i in range(2)])
print(x.shape)