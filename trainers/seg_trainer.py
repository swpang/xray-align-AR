import torch
import math
from models.seg_model import SegModel
from trainers.base_trainer import BaseTrainer
from util.utils import learning_rate_scheduler


class SegTrainer(BaseTrainer):
    def __init__(self):
        super(SegTrainer, self).__init__()

    def initialize(self, opt):
        self.opt = opt  # This line is added On Feb 21
        self.seg_model = SegModel()
        self.netG, self.netD = self.seg_model.initialize(opt)
        if len(opt['gpu_ids']):
            model = torch.nn.DataParallel(self.seg_model, device_ids=opt['gpu_ids'])

        print("model [%s] was created" % self.seg_model.name)

        if opt['isTrain']:
            self.optimizer_G, self.optimizer_D = self.seg_model.set_optimizers(self.netG, self.netD)
            self.old_lr = opt['lr']

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        G_losses, generated = self.seg_model(data, mode='generator')
        loss_G = sum(G_losses.values()).mean()
        loss_G.backward()
        self.optimizer_G.step()
        self.G_losses = G_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        D_losses = self.seg_model(data, mode='discriminator')
        loss_D = sum(D_losses.values()).mean()
        loss_D.backward()
        self.optimizer_D.step()
        self.D_losses = D_losses

    def get_latest_losses(self):
        return {**self.G_losses, **self.D_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.seg_model.save(epoch)

    # Update Learning rate function
    def update_learning_rate(self, epoch):
        new_lr_G, new_lr_D, new_lr = learning_rate_scheduler(self.opt, self.old_lr, epoch)
        if new_lr != self.old_lr:
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_G
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_D
            print('update generator learning rate: %f -> %f' % (self.old_lr / 2, new_lr_G))
            print('update discriminator learning rate: %f -> %f' % (self.old_lr * 2, new_lr_D))
            self.old_lr = new_lr