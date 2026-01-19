import torch

assert torch.cuda.is_available(), "CUDA is not available. You are running CPU-only."
device = torch.device("cuda:0")

# Example tensors
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

z = x @ y  # runs on GPU
print("z device:", z.device)
