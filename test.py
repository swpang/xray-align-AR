import argparse
import os

import torch
import yaml
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
from PIL import Image

from data.test_datasets import VITONDataset, VITONDataLoader
from networks.generators import SegGenerator, GMM, ALIASGenerator
from util.utils import gen_noise, load_checkpoint, save_images, tensor2label, tensor2im

#TODO: incorporate trainers into inference code

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--test_model', type=str, required=True, help='seg, gmm, alias, all')
    parser.add_argument('--config', '--c', type=str, default='./configs/config_test.yaml')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e. g. 0  0,1,2,  0,2. use -1 for CPU')
    parser.add_argument('--which_epoch', type=str, default='latest', help='latest or multiples of 50')

    opt = parser.parse_args()
    return opt

def test(opt, seg, gmm, alias):
    seg_gen = True if seg is not None else False
    gmm_gen = True if gmm is not None else False
    alias_gen = True if alias is not None else False
    save_mid_imgs = opt['save_mid_results']

    up = nn.Upsample(size=(opt['load_height'], opt['load_width']), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            if not seg_gen:
                parse_map = inputs['parse_map'].cuda()
            pose = inputs['pose'].cuda()
            c = inputs['cloth']['unpaired'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()
            c_kp = inputs['cloth_keypoints']['unpaired'].cuda()

            if seg_gen is True:
                # Part 1. Segmentation generation
                parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 256), mode='bilinear')
                pose_down = F.interpolate(pose, size=(256, 256), mode='bilinear')
                c_masked_down = F.interpolate(c * cm, size=(256, 256), mode='bilinear')
                cm_down = F.interpolate(cm, size=(256, 256), mode='bilinear')
                seg_input = torch.cat((parse_agnostic_down, pose_down, gen_noise(c_masked_down.size()).cuda()), dim=1)

                parse_pred_down = seg(seg_input)
                parse_pred = gauss(up(parse_pred_down))
                parse_pred = parse_pred.argmax(dim=1)[:, None]

                if save_mid_imgs:
                    pil_image = Image.fromarray(tensor2label(parse_pred[0], 13))
                    if not os.path.exists(os.path.join(opt['save_dir'], 'seggen')):
                        os.mkdir(os.path.join(opt['save_dir'], 'seggen'))
                    pil_image.save(os.path.join(opt['save_dir'], 'seggen')+'/seg_' + img_names[0].replace('.jpg', '') + '_' + c_names[0])

                parse_old = torch.zeros(parse_pred.size(0), 13, opt['load_height'], opt['load_width'], dtype=torch.float).cuda()
                parse_old.scatter_(1, parse_pred, 1.0)

                labels = {
                    0:  ['background',  [0]],
                    1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                    2:  ['upper',       [3]],
                    3:  ['hair',        [1]],
                    4:  ['left_arm',    [5]],
                    5:  ['right_arm',   [6]],
                    6:  ['noise',       [12]]
                }
                parse = torch.zeros(parse_pred.size(0), 7, opt['load_height'], opt['load_width'], dtype=torch.float).cuda()
                for j in range(len(labels)):
                    for label in labels[j][1]:
                        parse[:, j] += parse_old[:, label]

            if gmm_gen is True:
                # Part 2. Clothes Deformation
                agnostic_gmm = F.interpolate(img_agnostic, size=(256, 256), mode='bicubic')
                cloth_gmm = F.interpolate(c_kp, size=(256, 256), mode='bicubic')
                if seg_gen is True:
                    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 256),mode='bicubic')
                else:
                    parse_cloth_gmm = F.interpolate(parse_map[:, 3:4], size=(256, 256), mode='bicubic')
                pose_gmm = F.interpolate(pose, size=(256, 256), mode='bicubic')
                c_gmm = F.interpolate(c * cm, size=(256, 256), mode='bicubic')
                gmm_input_A = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
                gmm_input_B = torch.cat((c_gmm, cloth_gmm), dim=1)

                _, warped_grid = gmm(gmm_input_A, gmm_input_B)
                warped_c = F.grid_sample(c*cm, warped_grid, padding_mode='border')
                warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

                if save_mid_imgs:
                    warped_c_image = Image.fromarray(tensor2im(warped_c[0]))
                    warped_c_image.save(os.path.join(opt['save_dir'], 'warped_c')+'/warped_c_' + img_names[0].replace('_cnt_0_background_removed.jpg', '') + '_' + c_names[0])
                    warped_cm_image = Image.fromarray(tensor2im(warped_cm[0], normalize=False))
                    warped_cm_image.save(os.path.join(opt['save_dir'], 'warped_c')+'/warped_cm_' + img_names[0].replace('_cnt_0_background_removed.jpg', '') + '_' + c_names[0])

            if alias_gen:
                # Part 3. Try-on synthesis
                misalign_mask = parse[:, 2:3] - warped_cm
                misalign_mask[misalign_mask < 0.0] = 0.0
                parse_div = parse
                parse_div[:, 2:3] -= misalign_mask

                output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

                unpaired_names = []
                print(img_names, c_names)
                for img_name, c_name in zip(img_names, c_names):
                    unpaired_names.append('{}_{}'.format(img_name.replace('_cnt_0_background_removed.jpg', ''), c_name))

                save_images(output, unpaired_names, opt['save_dir'])

            if (i + 1) % opt['display_freq'] == 0:
                print("step: {}".format(i + 1))

def main():
    config = get_opt()
    opt = yaml.load(open(config.config), Loader=yaml.FullLoader)

    # Add command line arguments to config file options
    opt['name'] = config.name
    opt['gpu_ids'] = config.gpu_ids
    opt['test_model'] = config.test_model
    opt['which_epoch'] = config.which_epoch

    str_ids = opt['gpu_ids'].split(',')
    opt['gpu_ids'] = []
    for str_id in str_ids:
        idx = int(str_id)
        if idx >= 0:
            opt['gpu_ids'].append(idx)

    # set gpu ids
    if len(opt['gpu_ids']) > 0:
        torch.cuda.set_device(opt['gpu_ids'][0])

    if not os.path.exists(opt['save_dir']):
        os.makedirs(opt['save_dir'])

    seg, gmm, alias = None, None, None

    if opt['test_model'] == 'seg' or opt['test_model'] == 'all':
        seg = SegGenerator(opt, input_nc=opt['semantic_nc'] + 6, output_nc=opt['semantic_nc'])
        print(seg)
        load_checkpoint(seg, os.path.join(opt['checkpoint_dir'], opt['seg_checkpoint']))
        seg.cuda().eval()
    if opt['test_model'] == 'gmm' or opt['test_model'] == 'all':
        gmm = GMM(opt, inputA_nc=7, inputB_nc=6)
        print(gmm)
        load_checkpoint(gmm, os.path.join(opt['checkpoint_dir'], opt['gmm_checkpoint']))
        gmm.cuda().eval()
    if opt['test_model'] == 'alias' or opt['test_model'] == 'all':
        opt['semantic_nc'] = 7
        alias = ALIASGenerator(opt, input_nc=9)
        opt['semantic_nc'] = 13
        print(alias)
        load_checkpoint(alias, os.path.join(opt['checkpoint_dir'], opt['alias_checkpoint']))
        alias.cuda().eval()

    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()