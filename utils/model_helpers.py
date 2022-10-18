import pandas as pd
import random
import os
import torch
import numpy as np
from torch.nn import init
from PIL import Image

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
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def get_saving_indices(len, max):
    return random.sample(range(0, max-1), len)

def init_checkpoint_dir(save_dir, epoch):
    folder = os.path.join(save_dir, 'epoch_%s' % epoch)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def assemble_stats(iters, loss_G, loss_D):
    return {'iter' : np.full(len(loss_G), iters),
            'loss_G': loss_G,
            'loss_D': loss_D}

def save_stats(path, iters, loss_G, loss_D):
    save_path = os.path.join(path, 'losses.csv')
    stats = assemble_stats(iters, loss_G, loss_D)
    pd.DataFrame(stats).to_csv(save_path, index=False)
    print('stats saved to %s' % save_path)

def tensor_to_img(input_image, imtype=np.uint8):
    """"Convert a Tensor array into a numpy image array."""
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    return image_numpy.astype(imtype)