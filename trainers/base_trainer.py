import math

class BaseTrainer:
    def __init__(self):
        super(BaseTrainer, self).__init__()
        self.opt = None

        self.seg_model = None
        self.gmm_model = None
        self.alias_model = None

        self.netG = None
        self.netD = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer = None
        self.generated = None
        self.old_lr = None
        self.G_losses = None
        self.D_losses = None
        self.Losses = None



