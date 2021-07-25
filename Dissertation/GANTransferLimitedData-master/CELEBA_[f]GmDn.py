import argparse
import os, sys
from os import path
import time
import copy
import torch
from torch import nn
import numpy as np
import random
# from torchsummary import summary
import shutil
import scipy.io as sio


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(999)


from gan_training import utils
from gan_training.train import Trainer, update_average
from gan_training.toggle_ImageNet import toggle_grad_G, toggle_grad_D, model_equal_part_G, model_equal_part_D
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (
    load_config, build_models, build_optimizers, build_lr_scheduler, build_models_PRE,
)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total=', total_num, 'Trainable=', trainable_num, 'fixed=', total_num-trainable_num)


def load_part_model(m_fix, m_ini):
    dict_fix = m_fix.state_dic()
    dict_ini = m_ini.state_dic()

    dict_fix = {k: v for k, v in dict_fix.items() if k in dict_ini and k.find('embedding')==-1 and k.find('fc') == -1}
    dict_ini.update(dict_fix)
    m_ini.load_state_dict(dict_ini)
    return m_ini


def model_equal_all(model, dict):
    model_dict = model.state_dict()
    model_dict.update(dict)
    model.load_state_dict(model_dict)
    return model


def model_equal_part(model, dict_all):
    model_dict = model.state_dict()
    dict_fix = {k: v for k, v in dict_all.items() if k in model_dict and k.find('embedding') == -1 and k.find('fc') == -1}
    model_dict.update(dict_fix)
    model.load_state_dict(model_dict)
    return model

''' ===================--- Set the traning mode ---==========================
DATA: going to train
DATA_FIX: used as a fixed pre-trained model
G_Layer_FIX, D_Layer_FIX: number of layers to fix
============================================================================='''
DATA = 'CELEBA'
DATA_FIX = 'ImageNet'
Num_epoch = 50
NNN = 10000
config_path = './configs/celebA.yaml'# you need to change it
main_path = './' # you need to change it
image_path = "./data/CelebA/" # you need to change it
image_test = "./data/CelebA/" # you need to change it
out_path = main_path+'/results_celebA_GmPn/'# you need to change it
load_dir = './pretrained_model/' # put the model pretrained on ImageNet here


for choose in range(3):

    if choose == 0:
        G_Layer_FIX = -4
        D_Layer_FIX = 2
    elif choose == 1:
        G_Layer_FIX = -2
        D_Layer_FIX = 2
    elif choose == 2:
        G_Layer_FIX = -5
        D_Layer_FIX = 2

    config = load_config(config_path, 'configs/default.yaml')

    config['generator']['layers'] = G_Layer_FIX
    config['discriminator']['layers'] = D_Layer_FIX
    config['data']['train_dir'] = image_path
    config['data']['test_dir'] = image_test
    config['generator']['name'] = 'resnet2'
    config['discriminator']['name'] = 'resnet2'
    #config['generator']['embed_size'] = 256
    config['z_dist']['dim'] = 256

    config['training']['out_dir'] = out_path + 'G_%d_D_%d/'%(-G_Layer_FIX, D_Layer_FIX)
    if not os.path.isdir(config['training']['out_dir']):
        os.makedirs(config['training']['out_dir'])

    if 1:
        # Short hands
        batch_size = config['training']['batch_size']
        d_steps = config['training']['d_steps']
        restart_every = config['training']['restart_every']
        inception_every = config['training']['inception_every']
        save_every = config['training']['save_every']
        backup_every = config['training']['backup_every']
        sample_nlabels = config['training']['sample_nlabels']

        out_dir = config['training']['out_dir']
        checkpoint_dir = path.join(out_dir, 'chkpts')

        # Create missing directories
        if not path.exists(out_dir):
            os.makedirs(out_dir)
        if not path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        shutil.copyfile(sys.argv[0], out_dir + '/training_script.py')

        # Logger
        checkpoint_io = CheckpointIO(
            checkpoint_dir=checkpoint_dir
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset
        train_dataset, nlabels = get_dataset(
            name=config['data']['type'],
            data_dir=config['data']['train_dir'],
            size=config['data']['img_size'],
            lsun_categories=config['data']['lsun_categories_train']
        )
        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=config['training']['nworkers'],
                shuffle=True, pin_memory=True, sampler=None, drop_last=True
        )

        test_dataset, nlabels = get_dataset(
            name=config['data']['type'],
            data_dir=config['data']['test_dir'],
            size=config['data']['img_size'],
            lsun_categories=config['data']['lsun_categories_train']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=True, pin_memory=True, sampler=None, drop_last=True
        )

        # Number of labels
        nlabels = min(nlabels, config['data']['nlabels'])
        sample_nlabels = min(nlabels, sample_nlabels)

        # Create models
        ''' --------- Choose the fixed layer ---------------'''
        generator, discriminator = build_models(config)
        dict_G = torch.load(load_dir + DATA_FIX + 'Pre_generator')
        generator = model_equal_part_G(generator, dict_G, fix_layer=G_Layer_FIX)
        dict_D = torch.load(load_dir + DATA_FIX + 'Pre_discriminator')
        discriminator = model_equal_part_D(discriminator, dict_D, fix_layer=D_Layer_FIX)

        for param in generator.parameters():
            param.requires_grad = False
        for param in discriminator.parameters():
            param.requires_grad = False

        toggle_grad_G(generator, True, G_Layer_FIX)
        toggle_grad_D(discriminator, True, D_Layer_FIX)

        # Put models on gpu if needed
        generator, discriminator = generator.to(device), discriminator.to(device)
        g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

        # summary(generator, input_size=[(256,), (1,)])
        # summary(discriminator, input_size=[(3, 128, 128), (1,)])

        # Register modules to checkpoint
        checkpoint_io.register_modules(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
        )




        # Logger
        logger = Logger(
            log_dir=path.join(out_dir, 'logs'),
            img_dir=path.join(out_dir, 'imgs'),
            monitoring=config['training']['monitoring'],
            monitoring_dir=path.join(out_dir, 'monitoring')
        )

        # Distributions
        ydist = get_ydist(nlabels, device=device)
        zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                          device=device)

        # Save for tests
        ntest = batch_size
        x_real, ytest = utils.get_nsamples(train_loader, ntest)
        ytest.clamp_(None, nlabels-1)
        ytest = ytest.to(device)
        ztest = zdist.sample((ntest,)).to(device)
        utils.save_images(x_real, path.join(out_dir, 'real.png'))

        # Test generator
        if config['training']['take_model_average']:
            generator_test = copy.deepcopy(generator)
            checkpoint_io.register_modules(generator_test=generator_test)
        else:
            generator_test = generator

        # Evaluator
        # NNN = 8000
        x_real, _ = utils.get_nsamples(test_loader, NNN)
        evaluator = Evaluator(generator_test, zdist, ydist,
                              batch_size=batch_size, device=device,
                              fid_real_samples=x_real, inception_nsamples=NNN, fid_sample_size=NNN)
        # Train
        tstart = t0 = time.time()


        it = -1
        epoch_idx = -1

        # Reinitialize model average if needed
        if (config['training']['take_model_average']
                and config['training']['model_average_reinit']):
            update_average(generator_test, generator, 0.)

        # Learning rate anneling
        g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
        d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

        # Trainer
        trainer = Trainer(
            generator, discriminator, g_optimizer, d_optimizer,
            gan_type=config['training']['gan_type'],
            reg_type=config['training']['reg_type'],
            reg_param=config['training']['reg_param'],
            D_fix_layer=config['discriminator']['layers']
        )

    # Training loop
    print('Start training...')
    save_dir = config['training']['out_dir'] + '/models/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    FLAG = 500
    get_parameter_number(generator)
    get_parameter_number(discriminator)

    inception_mean_all = []
    inception_std_all = []
    fid_all = []

    for epoch_idx in range(Num_epoch):
        # epoch_idx += 1
        print('Start epoch %d...' % epoch_idx)

        for x_real, y in train_loader:
            it += 1
            g_scheduler.step()
            d_scheduler.step()

            d_lr = d_optimizer.param_groups[0]['lr']
            g_lr = g_optimizer.param_groups[0]['lr']
            # logger.add('learning_rates', 'discriminator', d_lr, it=it)
            # logger.add('learning_rates', 'generator', g_lr, it=it)

            x_real, y = x_real.to(device), y.to(device)
            y.clamp_(None, nlabels-1)

            # Generators updates
            z = zdist.sample((batch_size,))
            gloss, x_fake = trainer.generator_trainstep(y, z, FLAG + 1.0)
            FLAG = FLAG * 0.9995

            if config['training']['take_model_average']:
                update_average(generator_test, generator,
                               beta=config['training']['model_average_beta'])

            # Discriminator updates
            dloss, reg = trainer.discriminator_trainstep(x_real, y, x_fake)

            with torch.no_grad():
                # (i) Sample if necessary
                if (it % config['training']['sample_every']) == 0:
                    d_fix, d_update = discriminator.conv_img.weight[1, 1, 1, 1], discriminator.fc.weight[0, 1]
                    g_fix, g_update = generator.conv_img.weight[1, 1, 1, 1], 0.0

                    print('[epoch %0d, it %4d] g_loss = %.4f, d_loss = %.4f, reg=%.4f, d_fix=%.4f, d_update=%.4f, g_fix=%.4f, g_update=%.4f'
                          % (epoch_idx, it, gloss, dloss, reg, d_fix, d_update, g_fix, g_update))
                    # print('Creating samples...')
                    x, _ = evaluator.create_samples(ztest, ytest)
                    logger.add_imgs(x, 'all', it)

                # (ii) Compute inception if necessary
                if inception_every > 0 and ((it + 1) % 5000) == 0:
                    inception_mean, inception_std, fid = evaluator.compute_inception_score()
                    inception_mean_all.append(inception_mean)
                    inception_std_all.append(inception_std)
                    fid_all.append(fid)
                    print('test it %d: IS: mean %.2f, std %.2f, FID: mean %.2f, time: %2f' % (
                        it, inception_mean, inception_std, fid, time.time() - tstart))

                    FID = np.stack(fid_all)
                    Inception_mean = np.stack(inception_mean_all)
                    Inception_std = np.stack(inception_std_all)
                    sio.savemat(config['training']['out_dir'] + DATA + 'base_FID_IS.mat', {'FID': FID,
                                                                      'Inception_mean': Inception_mean,
                                                                      'Inception_std': Inception_std})

                # (iii) Backup if necessary
                if ((it + 1) % backup_every) == 0:
                    print('Saving backup...')
                    # checkpoint_io.save('model_%08d.pt' % it, it=it)

                    TrainModeSave = DATA + '_%08d_' % it
                    torch.save(generator_test.state_dict(), save_dir + TrainModeSave + 'Pre_generator')
                    torch.save(discriminator.state_dict(), save_dir + TrainModeSave + 'Pre_discriminator')

