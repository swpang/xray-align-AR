import torch
from torch import nn
from torch.nn import functional as F

from networks.cnn_networks import VGG19
from util.tps_grid_gen import TPSGridGen


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if self.gan_mode == 'ls':
            pass
        elif self.gan_mode == 'original':
            pass
        elif self.gan_mode == 'w':
            pass
        elif self.gan_mode == 'hinge':
            pass
        else:
            raise ValueError('gan_mode {} not implemented'.format(self.gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls': # mean squared loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                if isinstance(input_i, list):
                    pred = input_i[-1]
                else:
                    pred = input_i
                loss_tensor = self.loss(pred, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class ConstraintLoss(nn.Module):
    def __init__(self, opt):
        super(ConstraintLoss, self).__init__()
        self.opt = opt

    def get_row(self, coord, num):
        sec_dic=[]
        for j in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for i in range(num - 1):
                differ=(coord[:, j * num + i + 1, :] - coord[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ - buffer)
                    sec_dic.append(second_dif)

                buffer=differ
                sum+=second_dif
        return torch.stack(sec_dic,dim=1)

    def get_col(self,coor,num):
        sec_dic=[]
        for i in range(num):
            sum = 0
            buffer = 0
            flag = False
            max = -1
            for j in range(num - 1):
                differ = (coor[:, (j+1) * num + i , :] - coor[:, j * num + i, :]) ** 2
                if not flag:
                    second_dif = 0
                    flag = True
                else:
                    second_dif = torch.abs(differ-buffer)
                    sec_dic.append(second_dif)
                buffer = differ
                sum += second_dif
        return torch.stack(sec_dic,dim=1)

    def grad_row(self, coor, num):
        sec_term = []
        for j in range(num):
            for i in range(1, num - 1):
                x0, y0 = coor[:, j * num + i - 1, :][0]
                x1, y1 = coor[:, j * num + i + 0, :][0]
                x2, y2 = coor[:, j * num + i + 1, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def grad_col(self, coor, num):
        sec_term = []
        for i in range(num):
            for j in range(1, num - 1):
                x0, y0 = coor[:, (j - 1) * num + i, :][0]
                x1, y1 = coor[:, j * num + i, :][0]
                x2, y2 = coor[:, (j + 1) * num + i, :][0]
                grad = torch.abs((y1 - y0) * (x1 - x2) - (y1 - y2) * (x1 - x0))
                sec_term.append(grad)
        return sec_term

    def forward(self, theta):
        row = self.get_row(theta, self.opt['grid_size'])
        col = self.get_col(theta, self.opt['grid_size'])
        rg_loss = sum(self.grad_row(theta, self.opt['grid_size']))
        cg_loss = sum(self.grad_col(theta, self.opt['grid_size']))
        rg_loss = torch.max(rg_loss, torch.tensor(0.02).cuda())
        cg_loss = torch.max(cg_loss, torch.tensor(0.02).cuda())
        rx, ry, cx, cy = torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda() \
            , torch.tensor(0.08).cuda(), torch.tensor(0.08).cuda()
        row_x, row_y = row[:, :, 0], row[:, :, 1]
        col_x, col_y = col[:, :, 0], col[:, :, 1]
        rx_loss = torch.max(rx, row_x).mean()
        ry_loss = torch.max(ry, row_y).mean()
        cx_loss = torch.max(cx, col_x).mean()
        cy_loss = torch.max(cy, col_y).mean()

        return rx_loss + ry_loss + cx_loss + cy_loss + rg_loss + cg_loss

class AlignmentLoss(nn.Module):
    def __init__(self, opt):
        super(AlignmentLoss, self).__init__()
        self.opt = opt
        self.tps = TPSGridGen(self.opt)

    def forward(self, theta, pose_kp, img_kp, c_kp):
        self.tps.apply_transformation(theta, c_kp)
        return loss