import torch

def is_cuda_available():
    return torch.cuda.is_available()

if __name__ == "__main__":
    if is_cuda_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")