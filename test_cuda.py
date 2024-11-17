import torch

print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check the CUDA version PyTorch is built for
print(torch.cuda.is_available())  # Check if CUDA is available
