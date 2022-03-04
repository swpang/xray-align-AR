import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from models.base_model import BaseModel
from networks.generators import GMM
from networks.loss import ConstraintLoss, AlignmentLoss
from util.utils import ImagePool


class GmmModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = 'VitonHD_GMMModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True

        self.options = self.opt['gmm_options']

        #################### DEFINE NETWORKS ####################

        # Define Networks
        # Generator network
        self.netG = GMM(opt, inputA_nc=7, inputB_nc=6)

        print('---------- Networks initialized -------------')

        #################### LOAD NETWORKS ####################

        if not self.isTrain or opt['continue_train'] or opt['load_pretrain']:
            pretrained_path = '' if not self.isTrain else opt['load_pretrain']
            self.load_network(self.netG, 'G', opt['which_epoch'], pretrained_path)

        #################### SET LOSS FUNCTIONS AND OPTIMIZERS ####################
        # TODO: why Imagepool is used - Need to understand
        if self.isTrain:
            if opt['pool_size'] > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt['pool_size'])
            self.old_lr = opt['lr']

            # Define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionConst = ConstraintLoss(opt)
            # self.criterionKP = AlignmentLoss(opt)
        return self.netG

    def calculate_loss(self, inputs):
        Losses = {} # L1, L2_const (TPS_Grid_constraints), KP
        Generated = {} # warped_c, warped_cm, theta, warped_grid, warped_kp

        theta, warped_grid = self.netG.forward(self.gmm_input, torch.cat((self.c_gmm, self.cloth_gmm), dim=1))

        warped_c = F.grid_sample(self.c * self.cm, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(self.cm, warped_grid, padding_mode='border')
        warped_kp = F.grid_sample(self.c_kp, warped_grid, padding_mode='border')

        theta_x = theta[:, :self.opt['grid_size'] * self.opt['grid_size']].unsqueeze(2)
        theta_y = theta[:, self.opt['grid_size'] * self.opt['grid_size']:].unsqueeze(2)
        theta = torch.cat((theta_x, theta_y), dim=2)

        Losses['L1'] = self.criterionL1(warped_c, self.im_c)
        # FIXME: No mention of L1 Mask Loss in VITON-HD paper! Is this necessary?
        # Losses['L1_mask'] = self.criterionL1(self.warped_cm, self.parse_map[:,3:4])
        Losses['L2_const'] = self.criterionConst(theta) * self.opt['lambda_const']
        Losses['KP'] = self.criterionL1(warped_kp, self.pose_kp) * self.opt['lambda_kp']

        Generated['warped_c'] = warped_c
        Generated['warped_cm'] = warped_cm
        Generated['theta'] = theta
        Generated['warped_grid'] = warped_grid
        Generated['warped_kp'] = warped_kp

        return Losses, Generated

    # Set Optimizer
    def set_optimizers(self, netG):
        params = list(netG.parameters())
        self.optimizer_G = torch.optim.Adam(
            params, lr=self.opt['lr'], betas=(self.options['beta1_G'], self.options['beta2_G']))

        return self.optimizer_G

    def preprocess_input(self, inputs):
        self.img_agnostic = inputs['img_agnostic'].cuda()
        self.img = inputs['img'].cuda()
        self.parse_map = inputs['parse_map'].cuda()
        self.pose = inputs['pose'].cuda()
        self.c = inputs['cloth']['unpaired'].cuda()
        self.cm = inputs['cloth_mask']['unpaired'].cuda()
        self.pose_kp = inputs['pose_keypoints'].cuda()
        self.c_kp = inputs['cloth_keypoints']['unpaired'].cuda()
        self.im_c = self.img * self.parse_map[:, 3:4]

        # Interpolate method default = nearest
        self.agnostic_gmm = F.interpolate(self.img_agnostic, size=(256, 256))
        self.parse_cloth_gmm = F.interpolate(self.parse_map[:, 3:4], size=(256, 256))
        self.c_gmm = F.interpolate(self.c * self.cm, size=(256, 256))
        self.pose_gmm = F.interpolate(self.pose, size=(256, 256))
        self.cloth_gmm = F.interpolate(self.c_kp, size=(256, 256))

        self.gmm_input = torch.cat((self.parse_cloth_gmm, self.pose_gmm, self.agnostic_gmm), dim=1)

    # Forward function for the entire network
    def forward(self, inputs, mode):
        self.preprocess_input(inputs)
        if mode == 'train':
            loss, generated = self.calculate_loss(self.gmm_input)
            return loss, generated
        elif mode == 'inference':
            with torch.no_grad():
                _, warped_grid =  self.netG.forward(inputs, self.c_gmm)
                warped_c = F.grid_sample(self.c * self.cm, warped_grid, padding_mode='border')
                warped_cm = F.grid_sample(self.cm, warped_grid, padding_mode='border')
            return warped_c, warped_cm
        else:
            raise ValueError('|mode| is invalid')

