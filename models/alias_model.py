import os
import torch

from .base_model import BaseModel
from networks.discriminators import define_D
from networks.generators import ALIASGenerator
from networks.loss import GANLoss, VGGLoss


class AliasModel(BaseModel):
    def __init__(self):
        super(AliasModel, self).__init__()
        self.name = 'VitonHD_AliasModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True

        self.options = opt['alias_options']

        #################### DEFINE NETWORKS ####################

        # Define Networks
        # Generator network
        opt['semantic_nc'] = 7
        self.netG = ALIASGenerator(opt, input_nc=9)

        # Discriminator network
        netD_input_nc = 10
        use_sigmoid = True if self.options['gan_mode'] is not 'ls' else False
        self.netD = define_D(netD_input_nc, opt['ndf'], opt['n_layers_D'], opt['norm'], use_sigmoid,
                                      opt['num_D'], getIntermFeat=False, gpu_ids=self.gpu_ids)
        print('---------- Networks initialized -------------')

        #################### LOAD NETWORKS ####################

        if self.isTrain:
            if opt['continue_train'] or opt['load_pretrain']:
                pretrained_path = os.path.join(opt['checkpoint_dir'], opt['name'])
                self.load_network(self.netG, 'G', opt['which_epoch'], pretrained_path)
                self.load_network(self.netD, 'D', opt['which_epoch'], pretrained_path)
        else:
            pretrained_path = os.path.join(opt['checkpoint_dir'], opt['name'])
            self.load_network(self.netG, 'G', opt['which_epoch'], pretrained_path)

        #################### SET LOSS FUNCTIONS AND OPTIMIZERS ####################
        # TODO:  Need to understand why Imagepool is used
        if self.isTrain:
            # Define loss functions
            self.criterionGAN = GANLoss(self.options['gan_mode'], tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss()

        return self.netG, self.netD

    def discriminate(self, input_semantics, real_image, fake_image):
        input_fake_concat = torch.cat((input_semantics, fake_image.detach()), dim=1)
        input_real_concat = torch.cat((input_semantics, real_image.detach()), dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
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
    def forward(self, inputs, mode):
        self.preprocess_input(inputs)

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(self.img)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(self.img)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_img = self.netG.forward(torch.cat((self.img_agnostic, self.pose, self.warped_c), dim=1), self.alias_parse,
                                     self.parse_div, self.misalign_mask)
            return fake_img
        else:
            raise ValueError('|mode| is invalid')

    def set_optimizers(self, netG, netD):
        # Initialize optimizers
        # Optimizers
        G_params = list(netG.parameters())
        D_params = list(netD.parameters())

        if self.opt['lr_update'] == 'TTUR':
            self.optimizer_G = torch.optim.Adam(G_params,
                lr=self.opt['lr'] / 2,
                betas=(self.options['beta1_G'], self.options['beta2_G']))
            self.optimizer_D = torch.optim.Adam(D_params,
                lr=self.opt['lr'] * 2,
                betas=(self.options['beta1_D'], self.options['beta2_D']))
        return self.optimizer_G, self.optimizer_D

    def preprocess_input(self, inputs):
        self.img = inputs['img'].cuda()
        self.img_agnostic = inputs['img_agnostic'].cuda()
        self.pose = inputs['pose'].cuda()
        self.parse_map = inputs['parse_map'].cuda()
        self.warped_c = inputs['warped_cloth']['unpaired'].cuda()
        self.warped_cm = inputs['warped_cloth_mask']['unpaired'].cuda()

        labels = {
            0: ['background', [0]],
            1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
            2: ['upper', [3]],
            3: ['hair', [1]],
            4: ['left_arm', [5]],
            5: ['right_arm', [6]],
            6: ['noise', [12]]
        }
        self.alias_parse = torch.zeros(self.parse_map.size(0), 7, self.opt['load_height'], self.opt['load_width'],
                                  dtype=torch.float).cuda()
        for j in range(len(labels)):
            for label in labels[j][1]:
                self.alias_parse[:, j] += self.parse_map[:, label]

        self.misalign_mask = self.alias_parse[:, 2:3] - self.warped_cm
        self.misalign_mask[self.misalign_mask < 0.0] = 0.0
        self.misalign_mask = self.misalign_mask.cuda()

        #self.parse_div = torch.cat((self.alias_parse, self.misalign_mask), dim=1)
        self.parse_div = self.alias_parse
        self.parse_div[:, 2:3] -= self.misalign_mask

    def compute_generator_loss(self, real_img):
        G_losses = {}

        fake_img = self.netG.forward(torch.cat((self.img_agnostic, self.pose, self.warped_c), dim=1), self.alias_parse,
                                     self.parse_div, self.misalign_mask)

        pred_fake, pred_real = self.discriminate(self.parse_div, real_img, fake_img)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        # GAN feature matching loss
        G_losses['FM'] = self.FloatTensor(1).fill_(0)
        feat_weights = 4.0 / (self.opt['n_layers_D'] + 1)
        D_weights = 1.0 / self.opt['num_D']
        for i in range(self.opt['num_D']):
            for j in range(len(pred_fake[i]) - 1):
                G_losses['FM'] += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * \
                           self.opt['lambda_feat']

        # VGG feature matching loss
        G_losses['percept'] = self.criterionVGG(fake_img, real_img) * self.opt['lambda_percept']

        return G_losses, fake_img

    def compute_discriminator_loss(self, real_img):
        D_losses = {}

        with torch.no_grad():
            fake_img = self.netG.forward(torch.cat((self.img_agnostic, self.pose, self.warped_c), dim=1), self.alias_parse,
                                         self.parse_div, self.misalign_mask)
            fake_img = fake_img.detach()
            fake_img.requires_grad_()

        pred_fake, pred_real = self.discriminate(self.parse_div, real_img, fake_img)

        D_losses['fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['real'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt['lambda_real']

        return D_losses

    # Save model function
    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
