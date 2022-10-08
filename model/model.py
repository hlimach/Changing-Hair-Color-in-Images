from .networks import Generator, NLayerDiscriminator
import torch.nn as nn

class HairColorGAN(object):
    def __init__(self, params, is_train=True):
        super().__init__()
        self.params = params
        self.is_train = is_train
        self.show_losses = []
        self.save_imgs = []

        self.gen = Generator(in_c=6, out_c=3, nf=64, params=self.params)

        if self.is_train:
            self.desc = NLayerDiscriminator(in_c=6, nf=64, params=self.params)

            self.GAN_loss = 1
            self.idt_loss = nn.L1Loss()
            self.cycle_loss = nn.L1Loss()

    