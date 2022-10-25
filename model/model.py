from .networks import Generator, NLayerDiscriminator
from .model_helpers import *
import torch.nn as nn
import torch
from PIL import Image
from torch.optim import lr_scheduler, Adam
import os
import numpy as np

class HairColorGAN(object):
    def __init__(self, params, i_max, is_train=True):
        super().__init__()
        self.params = params
        self.is_train = is_train
        self.i_max = i_max
        self.checkpoint = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen = Generator(in_c=6, out_c=3, params=self.params).to(self.device)
        init_weights(self.gen)
        self.model_names = ['gen']

        if self.is_train:
            self.disc = NLayerDiscriminator(in_c=6, params=self.params).to(self.device)
            init_weights(self.disc)
            self.model_names.append('disc')
            self.optimizers = []
            self.iter = 0
            self.setup_losses()
            self.setup_optimizers()
            self.setup_schedulers()

        if not self.is_train or self.params.continue_train:
            self.load_latest_checkpoint()

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
        self.schedulers = []
        for optimizer in self.optimizers:
            if self.params.lr_policy == 'linear':
                def lambda_rule(epoch):
                    lr_l = 1.0 - max(0, epoch + self.checkpoint - self.params.n_epochs) / float(self.params.n_epochs_decay + 1)
                    return lr_l
                self.schedulers.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule))
            elif self.params.lr_policy == 'step':
                self.schedulers.append(lr_scheduler.StepLR(optimizer, step_size=self.params.lr_decay_iters, gamma=0.1))

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
        target_tensor = get_target_label(True, pred, self.device)
        return self.crit_GAN(pred, target_tensor)
    
    def backward_D(self):
        real_output = self.disc(torch.cat((self.B, self.B_rgb), 1))
        real_loss = self.crit_GAN(real_output, get_target_label(True, real_output, self.device)) 
        fake_output = self.disc(torch.cat((self.fake_B, self.B_rgb), 1).detach())
        fake_loss = self.crit_GAN(fake_output, get_target_label(False, fake_output, self.device))
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

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']

    def refresh(self, epoch):
        self.iter_tracker = []
        self.loss_G_tracker = []
        self.loss_D_tracker = []
        self.epoch_dir = init_checkpoint_dir(self.params.save_dir, epoch)
        saved_count = len([f for f in os.listdir(self.epoch_dir) if f.endswith('.png')])
        self.save_indices = get_saving_indices(self.params.img_pool_size - saved_count, 
                                                self.iter, self.i_max)

    def update_trackers(self, i):
        self.iter_tracker.append(i)
        self.loss_G_tracker.append(self.loss_G.item())
        self.loss_D_tracker.append(self.loss_D.item())

    def get_stats(self):
        return {'lr': self.lr, 'loss_G': self.loss_G, 'loss_D': self.loss_D}
    
    def save_images(self, iter):
        a_save = tensor_to_img(self.A.detach())
        rgb_save = tensor_to_img(self.target_rgb.detach())
        fake_save = tensor_to_img(self.fake_B.detach())
        self.concat = np.concatenate((rgb_save, a_save, fake_save), axis = 1)
        self.image_pil = Image.fromarray(self.concat)
        path = os.path.join(self.epoch_dir, ('iter_%s.png') % (iter))
        self.image_pil.save(path)
    
    def save_logs(self, epoch):
        print('finished epoch ', epoch)
        save_stats(self.epoch_dir, self.iter_tracker, self.loss_G_tracker, self.loss_D_tracker)
        self.iter = 0
        
    def save_model(self, iter, type='best'):
        if type not in ['best', 'latest']:
            raise Exception("type of model should be either 'best' or 'latest'.")
        for model in self.model_names:
            filename = '%s_model_%s.pth' % (type, model)
            path = os.path.join(self.epoch_dir, filename)

            # delete previous [best or latest] model
            open(path, 'w').close()
            os.remove(path)
        
            # save current [best or latest] model
            net = getattr(self, model)
            torch.save({'iter': iter,
                        'model_state_dict': net.state_dict()
                        }, path)
            # print(('%s %s model saved to %s') % (type, model, path))

    def load_latest_checkpoint(self):
        folder = [f for f in os.listdir(self.params.save_dir) if 'epoch_' in f][-1]
        save_path = os.path.join(self.params.save_dir, folder)
        self.checkpoint = int(folder.split('_')[-1])
    
        for model in self.model_names:
            filename = 'latest_model_%s.pth' % (model)
            state_dict = torch.load(os.path.join(save_path, filename), map_location=self.device)
            net = getattr(self, model)
            net.load_state_dict(state_dict['model_state_dict'])
            print('successfully loaded: ', filename)

        self.iter = state_dict['iter'] + 1
        if self.iter == self.i_max:
            self.checkpoint += 1
            self.iter = 0
        print(('starting from epoch %s, iter %s') % (self.checkpoint, self.iter))