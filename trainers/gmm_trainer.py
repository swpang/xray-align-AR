import torch

from models.gmm_model import GmmModel
from trainers.base_trainer import BaseTrainer
from util.utils import learning_rate_scheduler


class GMMTrainer(BaseTrainer):
    def __init__(self):
        super(GMMTrainer, self).__init__()

    def initialize(self, opt):
        self.opt = opt
        self.gmm_model = GmmModel()
        self.netG = self.gmm_model.initialize(opt)
        if len(opt['gpu_ids']):
            model = torch.nn.DataParallel(self.gmm_model, device_ids=opt['gpu_ids'])

        print("model [%s] was created" % self.gmm_model.name)

        if opt['isTrain']:
            self.optimizer = self.gmm_model.set_optimizers(self.netG)
            self.old_lr = opt['lr']

    def run_forward_pass(self, data):
        self.optimizer.zero_grad()
        Losses, generated = self.gmm_model(data, mode='train' if self.opt['isTrain'] else 'inference')
        loss = sum(Losses.values()).mean()
        loss.backward()
        self.optimizer.step()
        self.Losses = Losses
        self.generated = generated

    def get_latest_losses(self):
        return self.Losses

    def get_latest_generated(self):
        return self.generated # warped_c, warped_cm, theta, warped_grid

    def save(self, epoch):
        self.gmm_model.save(epoch)

    # Update Learning rate function
    def update_learning_rate(self, epoch):
        new_lr_G, _, new_lr = learning_rate_scheduler(self.opt, self.old_lr, epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr_G
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr