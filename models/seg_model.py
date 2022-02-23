import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from networks.generators import SegGenerator
from networks.loss import GANLoss
from networks import discriminators


class SegModel(BaseModel):
    def __init__(self):
        super(SegModel, self).__init__()
        self.name = 'VitonHD_SegModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True

        self.options = opt['seg_options']

        #################### DEFINE NETWORKS ####################

        # Define Networks
        # Generator network
        self.netG = SegGenerator(opt, input_nc=self.opt['semantic_nc'] + 6, output_nc=self.opt['semantic_nc'])

        # Discriminator network
        netD_input_nc = 32 # 19 + 13
        use_sigmoid = True if self.options['gan_mode'] is not 'ls' else False
        self.netD = discriminators.define_D(netD_input_nc, self.opt['ndf'], self.opt['n_layers_D'], self.opt['norm'], use_sigmoid,
                                            self.opt['num_D'], False, gpu_ids=self.gpu_ids)
        print('---------- Networks initialized -------------')

        #################### LOAD NETWORKS ####################

        if not self.isTrain or self.opt['continue_train'] or self.opt['load_pretrain']:
            pretrained_path = '' if not self.isTrain else self.opt['load_pretrain']
            self.load_network(self.netG, 'G', self.opt['which_epoch'], pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', self.opt['which_epoch'], pretrained_path)

        #################### SET LOSS FUNCTIONS AND OPTIMIZERS ####################

        if self.isTrain:
            # Define loss functions
            self.criterionGAN = GANLoss(self.options['gan_mode'], tensor=self.Tensor)
            self.criterionCE = torch.nn.CrossEntropyLoss()

        return self.netG, self.netD

    # TODO: not sure if this is true. In BN, the fake and real images are recommended to be in
    # TODO: the same batch to avoid disparate statistics in the fake and real images. So in SPADE
    # TODO: both the fake and real images are fed into D at once.
    # -> I think this code is wrong.... input to the discriminator should be (X,S)
    def discriminate(self, input_semantics, real_image, fake_image):
        input_fake_concat = torch.cat((input_semantics, fake_image.detach()), dim=1)
        input_real_concat = torch.cat((input_semantics, real_image.detach()), dim=1)

        input_concat = torch.cat((input_fake_concat, input_real_concat), dim=0)
        discriminator_out = self.netD.forward(input_concat)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    @staticmethod
    def divide_pred(pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    # Forward function for the entire network
    '''
    S = segmentation map (ground truth)
    S_a = clothing-agnostic segmentation map (gt)
    S_hat = G_S(S_a, P, c)
    X = (S_a, P, c) (ground truth)
    L_cGAN = E_(X,S)[log(D(X,S)) + E_X[1 - log(D(X,S_hat))]
    '''
    def forward(self, inputs, mode):
        self.preprocess_input(inputs)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(self.parse_map_down)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(self.parse_map_down)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_seg = self.netG.forward(self.seg_input)
            return fake_seg
        else:
            raise ValueError('|mode| is invalid')

    def set_optimizers(self, netG, netD):
        # Initialize optimizers
        # Optimizers
        G_params = list(netG.parameters())
        D_params = list(netD.parameters())

        if self.opt['lr_update'] == 'TTUR':
            self.optimizer_G = torch.optim.Adam(G_params,
                lr=self.opt['lr'] / 2, betas=(self.options['beta1_G'], self.options['beta2_G']))
            self.optimizer_D = torch.optim.Adam(D_params,
                lr=self.opt['lr'] * 2, betas=(self.options['beta1_D'], self.options['beta2_D']))
        return self.optimizer_G, self.optimizer_D

    def preprocess_input(self, inputs):
        self.parse_agnostic = inputs['parse_agnostic'].cuda()
        self.parse = inputs['parse'].cuda()
        self.parse_map = inputs['parse_map'].cuda()
        self.pose = inputs['pose'].cuda()
        self.c = inputs['cloth']['unpaired'].cuda()
        self.cm = inputs['cloth_mask']['unpaired'].cuda()

        self.parse_agnostic_down = F.interpolate(self.parse_agnostic, size=(256, 256), mode='bilinear')
        self.parse_map_down = F.interpolate(self.parse_map, size=(256, 256), mode='bilinear')
        self.pose_down = F.interpolate(self.pose, size=(256, 256), mode='bilinear')
        self.c_masked_down = F.interpolate(self.c * self.cm, size=(256, 256), mode='bilinear')
        self.cm_down = F.interpolate(self.cm, size=(256, 256), mode='bilinear')

        # Updated
        self.seg_input = torch.cat((self.parse_agnostic_down, self.pose_down, self.c_masked_down), dim=1).cuda()

    def compute_generator_loss(self, real_seg):
        G_losses = {} # GAN, CE

        fake_seg = self.netG.forward(self.seg_input)
        pred_fake, pred_real = self.discriminate(self.seg_input, real_seg, fake_seg)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        G_losses['CE'] = self.criterionCE(fake_seg, self.parse) * self.opt['lambda_ce']

        return G_losses, fake_seg

    def compute_discriminator_loss(self, real_seg):
        D_losses = {} # Real, Fake

        with torch.no_grad():
            fake_seg = self.netG.forward(self.seg_input)
            fake_seg = fake_seg.detach()
            fake_seg.requires_grad_()

        pred_fake, pred_real = self.discriminate(self.seg_input, real_seg, fake_seg)

        D_losses['fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['real'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt['lambda_real']

        return D_losses

    # Save model function
    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)