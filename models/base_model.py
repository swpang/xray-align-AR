import os
import sys
import torch
from torch import nn
import torchgeometry as tgm

from util.utils import ImagePool


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.name = None

        self.img_agnostic = None
        self.img = None
        self.pose = None
        self.warped_c = None
        self.warped_cm = None
        self.parse = None
        self.parse_map = None
        self.parse_agnostic = None
        self.alias_parse = None
        self.misalign_mask = None
        self.parse_div = None
        self.c = None
        self.cm = None
        self.im_c = None
        self.agnostic_gmm = None
        self.parse_cloth_gmm = None
        self.c_gmm = None
        self.pose_gmm = None
        self.gmm_input = None
        self.parse_agnostic_down = None
        self.parse_map_down = None
        self.pose_down = None
        self.c_masked_down = None
        self.cm_down = None
        self.seg_input = None

        self.FloatTensor = None
        self.ByteTensor = None
        self.Tensor = None
        self.opt = None
        self.options = None
        self.isTrain = None
        self.save_dir = None
        self.gpu_ids = None
        self.fake_pool = None
        self.old_lr = None
        self.up = None
        self.gauss = None

        self.netG = None
        self.netD = None
        self.criterionGAN = None
        self.criterionVGG = None
        self.criterionFeat = None
        self.criterionL1 = None
        self.criterionConst = None
        self.criterionCE = None

        self.optimizer_G = None
        self.optimizer_D = None

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = self.opt['gpu_ids']
        self.isTrain = True if self.opt['mode'] is 'train' else False  # Train/Test mode
        self.save_dir = os.path.join(self.opt['checkpoint_dir'], self.opt['name'])
        self.up = nn.Upsample(size=(opt['load_height'], opt['load_width']), mode='bilinear')
        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
        self.gauss.cuda()

        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids[-1] is not -1 else torch.FloatTensor
        if opt['gpu_ids'][-1] is not -1:
            self.FloatTensor = torch.cuda.FloatTensor
            self.ByteTensor = torch.cuda.ByteTensor
        else:
            self.FloatTensor = torch.FloatTensor
            self.ByteTensor = torch.ByteTensor

        if self.isTrain:
            if self.opt['pool_size'] > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(self.opt['pool_size'])
            self.old_lr = self.opt['lr']

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        print(save_path)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise 'Generator must exist!'
        else:
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt['verbose']:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)
