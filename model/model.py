from .networks import Generator, NLayerDiscriminator
from utils.model_helpers import *
import torch.nn as nn
import torch
from torch.optim import lr_scheduler, Adam
import os

class HairColorGAN(object):
    def __init__(self, params, is_train=True):
        super().__init__()
        self.params = params
        self.is_train = is_train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = Generator(in_c=6, out_c=3, params=self.params)
        self.model_names = ['gen']

        if self.is_train:
            self.disc = NLayerDiscriminator(in_c=6, params=self.params)
            self.model_names.append('disc')
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
        """Defines criterions"""
        self.crit_GAN = nn.MSELoss()
        self.crit_idt = nn.L1Loss()
        self.crit_cycle = nn.L1Loss()

    def setup_optimizers(self):
        self.lr = self.params.lr
        self.optimizer_G = Adam(self.gen.parameters(), lr=self.lr, betas=(self.params.beta1, 0.999))
        self.optimizer_D = Adam(self.disc.parameters(), lr=self.lr, betas=(self.params.beta1, 0.999))
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
        self.loss_G = self.cycle_loss() + self.identity_loss() + self.gan_loss()
        self.loss_G.backward()

    def cycle_loss(self):
        """calculates loss between image A and fake A transformed back to original hair color"""
        bk_to_orig = self.gen(torch.cat((self.fake_B, self.A_rgb), 1))
        return self.crit_cycle(bk_to_orig, self.A) * self.params.lambda_cyc

    def identity_loss(self):
        """calculates loss between image B and image B generated using its original hair color"""
        if self.params.lambda_idt > 0:
            output = self.gen(torch.cat((self.B, self.B_rgb), 1))
            return self.crit_idt(output, self.B) * self.params.lambda_cyc * self.params.lambda_idt
        else:
            return 0
    
    def gan_loss(self):
        """calculates loss of discriminator judgement of generator output"""
        pred = self.disc(torch.cat((self.fake_B, self.target_rgb), 1))
        target_tensor = get_target_label(True, pred)
        return self.crit_GAN(pred, target_tensor)
    
    def backward_D(self):
        real_output = self.disc(torch.cat((self.B, self.B_rgb), 1))
        real_loss = self.crit_GAN(real_output, get_target_label(True, real_output)) 
        fake_output = self.disc(torch.cat((self.fake_B, self.B_rgb), 1).detach())
        fake_loss = self.crit_GAN(fake_output, get_target_label(False, fake_output))
        self.loss_D = (real_loss + fake_loss) / 2 # ([(D(y,h2)-1)^2]+[(D(G(x,h2),h2)-0)^2])/2
        self.loss_D.backward()
    
    def optimize_parameters(self):
        # forward step
        self.forward_G()
        
        # generator updates
        set_requires_grad([self.disc], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # discriminator updates
        set_requires_grad([self.disc], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def save_checkpoint(self, checkpoint):
        for model in self.model_names:
            filename = 'epoch_%s_%s.pth' % (checkpoint, model)
            save_path = os.path.join(self.save_dir, filename)
            net = getattr(self, model)

            torch.save({'epoch': checkpoint,
                        'model_state_dict': net.state_dict()
                        }, save_path)

## start from here
    def load_latest(self, checkpoint):
        for model in self.model_names:
            filename = 'epoch_%s_%s.pth' % (checkpoint, model)
            save_path = os.path.join(self.save_dir, filename)
            if os.path.exists(save_path):
                net = getattr(self, model)

                torch.save({'epoch': checkpoint,
                            'model_state_dict': net.state_dict()
                            }, save_path)