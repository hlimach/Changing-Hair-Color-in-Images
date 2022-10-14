import pandas as pd
import torch
from torch.nn import init

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

def init_weights(model, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def init_stats_file(epoch, stats):
    df=pd.DataFrame(columns=["epoch","train_iou","val_loss","val_iou"])