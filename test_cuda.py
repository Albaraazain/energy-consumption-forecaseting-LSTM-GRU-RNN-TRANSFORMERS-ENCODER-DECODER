import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print CUDA version
print("CUDA version:", torch.version.cuda)

# Print GPU information
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current Device ID:", torch.cuda.current_device())
