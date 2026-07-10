import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.rand(5000, 5000, device="cuda")
    y = torch.mm(x, x)

    print("✓ GPU computation successful!")
else:
    print("✗ GPU not available")