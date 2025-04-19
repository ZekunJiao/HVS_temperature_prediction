import torch

xx, xy = torch.meshgrid(torch.arange(10, dtype=torch.float32), torch.arange(10, dtype=torch.float32))
# normalize the coordinates
xx = xx / (10 - 1)
xy = xy / (10 - 1)
y_data = torch.stack([xy, xx])
print(y_data)