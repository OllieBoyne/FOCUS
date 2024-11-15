import torch

def get_device(device:str=None):
    """Get device for torch tensors"""
    if device is not None:
        return device

    elif torch.cuda.is_available():
        return 'cuda'

    elif torch.backends.mps.is_available():
        return 'mps'

    else:
        return 'cpu'