import torch
def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model