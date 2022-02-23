import torch
from torch import nn


class MaskNorm(nn.Module):
    def __init__(self, norm_nc):
        super(MaskNorm, self).__init__()
        self.norm_layer = nn.InstanceNorm2d(norm_nc, affine=False)

    def normalize_region(self, region, mask):
        b, c, h, w = region.size()

        num_pixels = mask.sum((2, 3), keepdim=True)  # size: (b, 1, 1, 1)
        num_pixels[num_pixels == 0] = 1
        mu = region.sum((2, 3), keepdim=True) / num_pixels  # size: (b, c, 1, 1)

        normalized_region = self.norm_layer(region + (1 - mask) * mu)
        return normalized_region * torch.sqrt(num_pixels / (h * w))

    def forward(self, x, mask):
        mask = mask.detach()
        normalized_foreground = self.normalize_region(x * mask, mask)
        normalized_background = self.normalize_region(x * (1 - mask), 1 - mask)
        return normalized_foreground + normalized_background


class ALIASNorm(nn.Module):
    def __init__(self, norm_type, norm_nc, label_nc):
        super(ALIASNorm, self).__init__()

        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))

        assert norm_type.startswith('alias')
        param_free_norm_type = norm_type[len('alias'):]
        if param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError(
                "'{}' is not a recognized parameter-free normalization type in ALIASNorm".format(param_free_norm_type)
            )

        nhidden = 128
        ks = 3 # kernel size
        pw = ks // 2 # padding width
        # gamma, beta functions for normalization
        self.conv_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.conv_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, seg, misalign_mask=None):
        # Part 1. Generate parameter-free normalized activations.
        b, c, h, w = x.size()
        noise = (torch.randn(b, w, h, 1).cuda() * self.noise_scale).transpose(1, 3)

        if misalign_mask is None:
            normalized = self.param_free_norm(x + noise)
        else:
            normalized = self.param_free_norm(x + noise, misalign_mask)

        # Part 2. Produce affine parameters conditioned on the segmentation map.
        actv = self.conv_shared(seg)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        # Apply the affine parameters.
        output = normalized * (1 + gamma) + beta
        return output