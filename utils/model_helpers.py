import torch

def get_target_label(is_real, pred):
    if is_real:
        target_tensor = torch.tensor(1.0)
    else:
        target_tensor = torch.tensor(0.0)
    return target_tensor.expand_as(pred)

def set_requires_grad(nets, requires_grad):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

