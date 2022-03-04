import argparse
import os
import time
import yaml
import torch
import numpy as np
from collections import OrderedDict

from data.train_datasets import VITONDataset, VITONDataLoader
from util.utils import tensor2im, tensor2label
from util.visualizer import Visualizer
from trainers.alias_trainer import ALIASTrainer
from trainers.gmm_trainer import GMMTrainer
from trainers.seg_trainer import SegTrainer

from util.cosine_scheduler import CosineScheduler


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--train_model', type=str, default='alias')
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--load_pretrain', type=str, default='')
    parser.add_argument('--which_epoch', type=str, default='latest')

    parser.add_argument('--config', '-c', type=str, default='configs/config.yaml')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    opt = parser.parse_args()
    return opt

def train_seg(opt):
    iter_path = os.path.join(opt['checkpoint_dir'], opt['name'], 'iter.txt')

    if opt['continue_train']:
        try:
            start_epoch, epoch_iter = np.loadtxt(
                iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    dataset_size = len(train_dataset)
    print('#training images = %d' % dataset_size)

    # Initialize Networks
    trainer = SegTrainer()
    trainer.initialize(opt)

    # Training Visualizer
    visualizer = Visualizer(opt)

    # Train related additional details
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt['display_freq']
    print_delta = total_steps % opt['print_freq']
    save_delta = total_steps % opt['save_latest_freq']

    for epoch in range(start_epoch, opt['niter'] + opt['niter_decay'] + 1):
        epoch_start_time = time.time()
        if epoch == start_epoch:
            epoch_iter %= dataset_size
        else:
            epoch_iter = 0

        for i, inputs in enumerate(train_loader.data_loader, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt['batch_size']
            epoch_iter += opt['batch_size']

            # whether to collect output images
            save_fake = total_steps % opt['display_freq'] == display_delta

            # train generator
            trainer.run_generator_one_step(inputs)

            # train discriminator
            trainer.run_discriminator_one_step(inputs)

            # print out errors
            losses = trainer.get_latest_losses()

            if total_steps % opt['print_freq'] == print_delta:
                errors = {k: v.data.item() if not isinstance(
                    v, int) else v for k, v in losses.items()}
                t = (time.time() - iter_start_time) / opt['print_freq']
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # display output images
            # TODO: save all images in a batch every few iterations
            if save_fake:
                visuals = OrderedDict([('real_label', tensor2label(inputs['parse_map'].data[0], 13)),
                                       ('fake_label', tensor2label(trainer.get_latest_generated().data[0], 13))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            # save latest model
            if total_steps % opt['save_latest_freq'] == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                trainer.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt['niter'] + opt['niter_decay'], iter_end_time - epoch_start_time))

        # save model for this epoch
        if epoch % opt['save_epoch_freq'] == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            trainer.save('latest')
            trainer.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # linearly decay learning rate after certain iterations
#        if epoch > opt['niter']:
#            trainer.update_learning_rate()
        trainer.update_learning_rate(epoch)  # Update lr for every single epoch


def train_gmm(opt):
    iter_path = os.path.join(opt['checkpoint_dir'], opt['name'], 'iter.txt')

    if opt['continue_train']:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    dataset_size = len(train_dataset)
    print('#training images = %d' % dataset_size)

    # Initialize Networks
    trainer = GMMTrainer()
    trainer.initialize(opt)

    # Training Visualizer
    visualizer = Visualizer(opt)

    # Train related additional details
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt['display_freq']
    print_delta = total_steps % opt['print_freq']
    save_delta = total_steps % opt['save_latest_freq']

    for epoch in range(start_epoch, opt['niter'] + opt['niter_decay'] + 1):
        epoch_start_time = time.time()
        if epoch == start_epoch:
            epoch_iter %= dataset_size
        else:
            epoch_iter = 0
        for i, inputs in enumerate(train_loader.data_loader, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt['batch_size']
            epoch_iter += opt['batch_size']

            # whether to collect output images
            save_fake = total_steps % opt['display_freq'] == display_delta

            # train GMM generator
            trainer.run_forward_pass(inputs)

            # print out errors
            losses = trainer.get_latest_losses()

            ############## Display results and errors ##########
            # print out errors
            if total_steps % opt['print_freq'] == print_delta:
                errors = {k: v.data.item() if not isinstance(
                    v, int) else v for k, v in losses.items()}
                t = (time.time() - iter_start_time)
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # display output images
            if save_fake:
                # TODO: get original images from inputs and generated images by get_latest_generated
                visuals = OrderedDict([('fake_cloth', tensor2im(trainer.get_latest_generated()['warped_c'].data[0])),
                                       ('real_cloth', tensor2im(inputs['cloth']['unpaired'].data[0])),
                                       ('fake_cloth_mask', tensor2im(trainer.get_latest_generated()['warped_cm'].data[0])),
                                       ('real_cloth_mask', tensor2im(inputs['cloth_mask']['unpaired'].data[0])),
                                       ('overlayed_cloth', tensor2im(trainer.get_latest_generated()['warped_c'].data[0] + inputs['img_agnostic']))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            # save latest model
            if total_steps % opt['save_latest_freq'] == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                trainer.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt['iter'] + opt['iter_decay'], iter_end_time - epoch_start_time))

        # save model for this epoch
        if epoch % opt['save_epoch_freq'] == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            trainer.save('latest')
            trainer.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # linearly decay learning rate after certain iterations
        if epoch > opt['niter']:
            trainer.update_learning_rate(epoch)

def train_alias(opt):
    iter_path = os.path.join(opt['checkpoint_dir'], opt['name'], 'iter.txt')

    if opt['continue_train']:
        try:
            start_epoch, epoch_iter = np.loadtxt(
                iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    dataset_size = len(train_dataset)
    print('#training images = %d' % dataset_size)

    # Initialize Networks
    trainer = ALIASTrainer()
    trainer.initialize(opt)

    # Training Visualizer
    visualizer = Visualizer(opt)

    # Train related additional details
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt['display_freq']
    print_delta = total_steps % opt['print_freq']
    save_delta = total_steps % opt['save_latest_freq']

    for epoch in range(start_epoch, opt['niter'] + opt['niter_decay'] + 1):
        epoch_start_time = time.time()
        if epoch == start_epoch:
            epoch_iter %= dataset_size
        else:
            epoch_iter = 0
        for i, inputs in enumerate(train_loader.data_loader, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt['batch_size']
            epoch_iter += opt['batch_size']

            # whether to collect output images
            save_fake = total_steps % opt['display_freq'] == display_delta

            # train generator
            trainer.run_generator_one_step(inputs)

            # train discriminator
            trainer.run_discriminator_one_step(inputs)

            # print out errors
            losses = trainer.get_latest_losses()

            if total_steps % opt['print_freq'] == print_delta:
                errors = {k: v.data.item() if not isinstance(
                    v, int) else v for k, v in losses.items()}
                t = (time.time() - iter_start_time)
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            # display output images
            # TODO: save all images in a batch every few iterations
            if save_fake and total_steps % opt['save_latest_freq'] == save_delta:
                visuals = OrderedDict([('real_img', tensor2im(inputs['img'].data[0])),
                                       ('fake_img', tensor2im(trainer.get_latest_generated().data[0])),
                                       ('warped_c', tensor2im(inputs['warped_cloth']['unpaired'].data[0])),
                                       ('img_agnostic', tensor2im(inputs['img_agnostic'].data[0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            # save latest model
            if total_steps % opt['save_latest_freq'] == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                trainer.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

        # end of epoch
        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt['niter'] + opt['niter_decay'], iter_end_time - epoch_start_time))

        # save model for this epoch
        if epoch % opt['save_epoch_freq'] == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            trainer.save('latest')
            trainer.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

        # linearly decay learning rate after certain iterations
        if epoch > opt['niter']:
            trainer.update_learning_rate()

def main():
    config = get_opt()
    opt = yaml.load(open(config.config), Loader=yaml.FullLoader)

    opt['name'] = config.name
    opt['train_model'] = config.train_model
    opt['mode'] = config.mode
    opt['continue_train'] = config.continue_train
    opt['load_pretrain'] = config.load_pretrain
    opt['which_epoch'] = config.which_epoch
    opt['config'] = config.config
    opt['gpu_ids'] = config.gpu_ids
    opt['isTrain'] = True if opt['mode'] is 'train' else False

    str_ids = opt['gpu_ids'].split(',')
    opt['gpu_ids'] = []
    for str_id in str_ids:
        idx = int(str_id)
        if idx >= 0:
            opt['gpu_ids'].append(idx)

    # set gpu ids
    if len(opt['gpu_ids']) > 0:
        torch.cuda.set_device(opt['gpu_ids'][0])

    if not os.path.exists(os.path.join(opt['checkpoint_dir'], opt['name'])):
        os.makedirs(os.path.join(opt['checkpoint_dir'], opt['name']))

    if opt['train_model'] == 'seg':
        train_seg(opt)
    elif opt['train_model'] == 'gmm':
        lr_update_orig = opt['lr_update']
        opt['lr_update'] = 'cosine'
        train_gmm(opt)
        opt['lr_update'] = lr_update_orig
    elif opt['train_model'] == 'alias':
        train_alias(opt)
    else:
        print(f"Undefined training model {opt['train_model']}")

if __name__ == '__main__':
    main()
