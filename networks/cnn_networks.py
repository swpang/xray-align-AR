import torch
from torch import nn
from torchvision.models import vgg19

from networks.base_network import BaseNetwork

class FeatureExtraction(BaseNetwork):
    def __init__(self, input_nc, ngf=64, num_layers=4, norm_layer=nn.BatchNorm2d):
        super(FeatureExtraction, self).__init__()

        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        for i in range(1, num_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1), nn.ReLU(), norm_layer(nf)]

        layers += [nn.Conv2d(nf, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(), norm_layer(512)]
        layers += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU()]

        self.model = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x):
        return self.model(x)

class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_size=6, norm_layer=nn.BatchNorm2d):
        super(FeatureRegression, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), norm_layer(512), nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), norm_layer(256), nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU()
        )
        self.linear = nn.Linear(64 * (input_nc // 16), output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, featureA, featureB):
        # Reshape features for matrix multiplication.
        b, c, h, w = featureA.size()
        featureA = featureA.permute(0, 3, 2, 1).reshape(b, w * h, c)
        featureB = featureB.reshape(b, c, h * w)

        # Perform matrix multiplication.
        corr = torch.bmm(featureA, featureB).reshape(b, w * h, h, w) # batch matrix multiplication
        return corr

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out