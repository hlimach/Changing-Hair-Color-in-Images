from .networks import Generator, NLayerDiscriminator
import torch.nn as nn
import torch
from torch.optim import lr_scheduler, Adam

class HairColorGAN(object):
    def __init__(self, params, is_train=True):
        super().__init__()
        self.params = params
        self.is_train = is_train
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_epochs = params.total_epochs
        self.checkpoint = params.checkpoint
        self.save_interval = params.save_interval

        self.gen = Generator(in_c=6, out_c=3, params=self.params)
        if self.is_train:
            self.desc = NLayerDiscriminator(in_c=6, params=self.params)
            self.setup_losses()
            self.setup_optimizers()
            self.setup_schedulers()

    def set_inputs(self, input):
        self.A = input['A'].to(self.device)
        self.B = input['B'].to(self.device)
        self.A_rgb = input['A_rgb'].to(self.device)
        self.B_rgb = input['B_rgb'].to(self.device)
        self.target_rgb = input['target_rgb'].to(self.device)

    def setup_losses(self):
        self.crit_GAN = nn.MSELoss()
        self.crit_idt = nn.L1Loss()
        self.crit_cycle = nn.L1Loss()

    def setup_optimizers(self):
        self.lr = self.params.lr
        self.optimizer_G = Adam(self.gen.parameters(), lr=self.lr, betas=(self.params.beta1, 0.999))
        self.optimizer_D = Adam(self.desc.parameters(), lr=self.lr, betas=(self.params.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
    
    def setup_schedulers(self):
        for optimizer in self.optimizers:
            if self.params.lr_policy == 'linear':
                def lambda_rule(epoch):
                    lr_l = 1.0 - max(0, epoch + self.params.epoch_count - self.params.n_epochs) / float(self.params.n_epochs_decay + 1)
                    return lr_l
                self.scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            elif self.params.lr_policy == 'step':
                self.scheduler = lr_scheduler.StepLR(optimizer, step_size=self.params.lr_decay_iters, gamma=0.1)

    def forward_G(self):
        self.fake_B = self.gen(torch.cat((self.A, self.target_rgb), 1))

    def backward_G(self):
        self.loss_G = self.get_cycle_loss() + self.get_identity_loss() + self.get_gan_loss()
        self.loss_G.backward()

    def get_cycle_loss(self):
        """calculates loss between image A and fake A transformed back to original hair color"""
        bk_to_orig = self.gen(torch.cat((self.fake_B, self.A_rgb), 1))
        return self.crit_cycle(bk_to_orig, self.A) * self.params.lambda_cyc

    def get_identity_loss(self):
        """calculates loss between image B and image B generated using its original hair color"""
        if self.params.lambda_idt > 0:
            output = self.gen(torch.cat((self.B, self.B_rgb), 1))
            return self.crit_idt(output, self.B) * self.params.lambda_cyc * self.params.lambda_idt
        else:
            return 0
    
    def get_gan_loss(self):
        pred = self.desc(torch.cat((self.fake_B, self.target_rgb), 1))
        target_tensor = self.get_target_label(True, pred)
        return self.crit_GAN(pred, target_tensor)
    
    def get_target_label(self, is_real, pred):
        if is_real:
            target_tensor = torch.tensor(1.0)
        else:
            target_tensor = torch.tensor(0.0)
        return target_tensor.expand_as(pred)