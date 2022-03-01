import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm

from networks.base_network import BaseNetwork
from networks.normalization import ALIASNorm
from networks.cnn_networks import FeatureRegression, FeatureCorrelation, FeatureExtraction
from util.tps_grid_gen import TPSGridGen


class SegGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        super(SegGenerator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=(3, 3), padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), norm_layer(64), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), norm_layer(128), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), norm_layer(256), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1), norm_layer(512), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1), norm_layer(1024), nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1), norm_layer(1024), nn.ReLU())

        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                 nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=1), norm_layer(512), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=1), norm_layer(512), nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1), norm_layer(512), nn.ReLU())

        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                 nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1), norm_layer(256), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1), norm_layer(256), nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1), norm_layer(256), nn.ReLU())

        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                 nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1), norm_layer(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1), norm_layer(128), nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1), norm_layer(128), nn.ReLU())

        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                 nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1), norm_layer(64), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1), norm_layer(64), nn.ReLU(),
                                   nn.Conv2d(64, output_nc, kernel_size=(3, 3), padding=1))

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.print_network()
        self.init_weights(opt['init_type'], opt['init_variance'])

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.drop(self.conv4(self.pool(conv3)))
        conv5 = self.drop(self.conv5(self.pool(conv4)))

        conv6 = self.conv6(torch.cat((conv4, self.up6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.up7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.up8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.up9(conv8)), 1))
        return self.sigmoid(conv9)

class GMM(nn.Module):
    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()

        self.extractionA = FeatureExtraction(inputA_nc, ngf=64, num_layers=4)
        self.extractionB = FeatureExtraction(inputB_nc, ngf=64, num_layers=4)
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=(opt['load_width'] // 32) * (opt['load_height'] // 32) ,
                                            output_size=2 * opt['grid_size']**2)
        self.gridGen = TPSGridGen(opt)

    def forward(self, inputA, inputB):
        featureA = F.normalize(self.extractionA(inputA), dim=1)
        featureB = F.normalize(self.extractionB(inputB), dim=1)
        corr = self.correlation(featureA, featureB)
        theta = self.regression(corr)

        warped_grid = self.gridGen(theta)
        return theta, warped_grid

class ALIASGenerator(BaseNetwork):
    def __init__(self, opt, input_nc):
        super(ALIASGenerator, self).__init__()
        self.num_upsampling_layers = opt['num_upsampling_layers']

        self.sh, self.sw = self.compute_latent_vector_size(opt)

        nf = opt['ngf']
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, kernel_size=(3, 3), padding=1)
        for i in range(1, opt['alias_options']['n_conv']):
            self.add_module('conv_{}'.format(i), nn.Conv2d(input_nc, 16, kernel_size=(3, 3), padding=1))

        self.head_0 = ALIASResBlock(opt, nf * 16, nf * 16)

        self.G_middle_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)
        self.G_middle_1 = ALIASResBlock(opt, nf * 16 + 16, nf * 16)

        self.up_0 = ALIASResBlock(opt, nf * 16 + 16, nf * 8)
        self.up_1 = ALIASResBlock(opt, nf * 8 + 16, nf * 4)
        self.up_2 = ALIASResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)
        self.up_3 = ALIASResBlock(opt, nf * 2 + 16, nf * 1, use_mask_norm=False)
        if self.num_upsampling_layers == 'most':
            self.up_4 = ALIASResBlock(opt, nf * 1 + 16, nf // 2, use_mask_norm=False)
            nf //= 2

        self.conv_img = nn.Conv2d(nf, 3, kernel_size=(3,3), padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        self.print_network()
        self.init_weights(opt['init_type'], opt['init_variance'])

    def compute_latent_vector_size(self, opt):
        if self.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif self.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif self.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError("opt.num_upsampling_layers '{}' is not recognized".format(self.num_upsampling_layers))

        sh = opt['load_height'] // 2**num_up_layers
        sw = opt['load_width'] // 2**num_up_layers
        return sh, sw

    def forward(self, x, seg, seg_div, misalign_mask):
        samples = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest') for i in range(8)]
        features = [self._modules['conv_{}'.format(i)](samples[i]) for i in range(8)]

        x = self.head_0(features[0], seg_div, misalign_mask)

        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, features[1]), 1), seg_div, misalign_mask)
        if self.num_upsampling_layers in ['more', 'most']:
            x = self.up(x)
        x = self.G_middle_1(torch.cat((x, features[2]), dim=1), seg_div, misalign_mask)

        x = self.up(x)
        x = self.up_0(torch.cat((x, features[3]), dim=1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_1(torch.cat((x, features[4]), dim=1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_2(torch.cat((x, features[5]), dim=1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, features[6]), dim=1), seg)
        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, features[7]), 1), seg)

        x = self.conv_img(self.relu(x))
        return self.tanh(x)

class ALIASResBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(ALIASResBlock, self).__init__()

        self.learned_shortcut = (input_nc != output_nc)
        middle_nc = min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=(3, 3), padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=(3, 3), padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=(1, 1), bias=False)

        subnorm_type = opt['norm_G']
        if subnorm_type.startswith('spectral'):
            subnorm_type = subnorm_type[len('spectral'):]
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        semantic_nc = opt['semantic_nc']
        if use_mask_norm:
            subnorm_type = 'aliasmask'

        self.norm_0 = ALIASNorm(subnorm_type, input_nc, semantic_nc)
        self.norm_1 = ALIASNorm(subnorm_type, middle_nc, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(subnorm_type, input_nc, semantic_nc)

        self.relu = nn.LeakyReLU(0.2)

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, misalign_mask))
        else:
            return x

    def forward(self, x, seg, misalign_mask=None):
        # maybe improvement by using a different more accurate interpolating algorithm? nearest interpolation isn't good!
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)

        dx = self.conv_0(self.relu(self.norm_0(x, seg, misalign_mask)))
        dx = self.conv_1(self.relu(self.norm_1(dx, seg, misalign_mask)))
        output = x_s + dx
        return output