import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import color_distillation.utils.transforms as T
import color_distillation.utils.ext_transforms as exT
from color_distillation import models, datasets
from color_distillation.models.color_cnn import ColorCNN
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.sampler import RandomSeqSampler
from color_distillation.utils.logger import Logger


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    data_path = os.path.expanduser('~/Data/') + args.dataset
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        num_class = 10 if args.dataset == 'cifar10' else 100
        pixsim_sample = 0.3
        img_size = 32

        train_trans = T.Compose([T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans,
                                         color_quantize=T.MedianCut())
            test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_trans,
                                        color_quantize=T.MedianCut())
        else:
            train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_trans,
                                          color_quantize=T.MedianCut())
            test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_trans,
                                         color_quantize=T.MedianCut())
    elif args.dataset == 'imagenet':
        num_class = 1000
        pixsim_sample = 0.1
        img_size = 224

        train_trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([T.Resize(256), T.CenterCrop(224), ])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(224, padding=28),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        train_set = datasets.ImageNet(data_path, split='train', transform=train_trans, color_quantize=T.MedianCut())
        test_set = datasets.ImageNet(data_path, split='val', transform=test_trans, color_quantize=T.MedianCut())
    elif args.dataset == 'style14mini':
        num_class = 14
        args.batch_size = 32
        pixsim_sample = 0.3
        img_size = 112

        train_trans = T.Compose([T.Resize(128), T.CenterCrop(112), T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([T.Resize(128), T.CenterCrop(112), ])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(112, padding=14),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans, color_quantize=T.MedianCut())
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans, color_quantize=T.MedianCut())
    elif args.dataset == 'stl10':
        num_class = 10
        # smaller batch size
        args.batch_size = 32
        pixsim_sample = 0.3
        img_size = 96

        train_trans = T.Compose([T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(96, padding=12),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        train_set = datasets.STL10(data_path, split='train', download=True, transform=train_trans,
                                   color_quantize=T.MedianCut())
        test_set = datasets.STL10(data_path, split='test', download=True, transform=test_trans,
                                  color_quantize=T.MedianCut())
    elif args.dataset == 'tiny200':
        num_class = 200
        pixsim_sample = 0.3
        img_size = 64

        train_trans = T.Compose([T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(64, padding=8),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans, color_quantize=T.MedianCut())
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans, color_quantize=T.MedianCut())
    elif args.dataset == 'voc_cls':
        num_class = 20
        args.batch_size = 32
        args.num_workers = 8
        args.log_interval = 200
        args.ce_ratio = 10
        pixsim_sample = 0.3
        data_path = os.path.expanduser('~/Data/pascal_VOC')
        img_size = 112

        train_trans = T.Compose([T.Resize(128), T.CenterCrop(112), T.RandomHorizontalFlip(), ])
        test_trans = T.Compose([T.Resize(128), T.CenterCrop(112), ])
        train_post_trans = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(112, padding=14),
                                      T.RandomRotation(degrees=15), T.RandomErasing()])

        train_set = datasets.VOCClassification(data_path, image_set='train', transform=train_trans,
                                               color_quantize=T.MedianCut())
        test_set = datasets.VOCClassification(data_path, image_set='val', transform=test_trans,
                                              color_quantize=T.MedianCut())
    elif args.dataset == 'voc_seg':
        num_class = 21
        # args.lr = 0.001
        args.batch_size = 8
        args.num_workers = 8
        args.arch = 'deeplab'
        args.log_interval = 2000
        pixsim_sample = 0.05
        crop_size = [160, 160]
        img_size = 160

        data_path = os.path.expanduser('~/Data/pascal_VOC')
        train_trans = exT.ExtCompose([exT.ExtResize(crop_size), exT.ExtRandomHorizontalFlip(), ])
        test_trans = exT.ExtCompose([exT.ExtResize(crop_size), ])
        train_post_trans = exT.ExtCompose([exT.ExtRandomHorizontalFlip(), exT.ExtRandomScale([0.5, 2.0]),
                                           exT.ExtRandomCrop(size=crop_size, pad_if_needed=True),
                                           exT.ExtRandomErasing()])

        train_set = datasets.VOCSegmentation(data_path, image_set='train', transforms=train_trans,
                                             color_quantize=T.MedianCut())
        test_set = datasets.VOCSegmentation(data_path, image_set='val', transforms=test_trans,
                                            color_quantize=T.MedianCut())
    else:
        raise Exception

    # network specific setting
    if args.arch == 'alexnet':
        if 'cifar' not in args.dataset:
            args.color_norm = 1

    kwargs = {'prefetch_factor': 1} if args.num_workers else {}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                             sampler=RandomSeqSampler(test_set) if args.dataset == 'style14mini' else None)

    logdir = f'{"DEBUG_" if is_debug else ""}' \
             f'{args.backbone}_agg{args.agg}_neck{args.bottleneck_channel}_colors{args.colors_channel}_topk{args.topk}_' \
             f'ce{args.ce_ratio}_kd{args.kd_ratio}_{args.cluster_loss}{args.pixsim_ratio}_sample{pixsim_sample}_recons{args.recons_ratio}_' \
             f'prcp{args.perceptual_ratio}_max{args.colormax_ratio}_conf{args.conf_ratio}_info{args.info_ratio}_' \
             f'jit{args.color_jitter}_norm{args.color_norm}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}/' \
        if not args.resume else f'resume_{args.resume}'  #
    logdir = f'logs/colorcnn/{args.dataset}/{args.arch}/{args.num_colors}colors/{logdir}'
    os.makedirs(f'{logdir}/imgs', exist_ok=True)
    copy_tree('./color_distillation', logdir + '/scripts/color_distillation')
    for script in os.listdir('.'):
        if script.split('.')[-1] == 'py':
            dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))
    print(logdir)

    # model
    def load_classifier(arch):
        classifier = models.create(arch, num_class).cuda()
        if args.dataset != 'imagenet':
            resume_fname = 'logs/grid/{}/{}/full_colors'.format(args.dataset, arch) + '/model.pth'
            classifier.load_state_dict(torch.load(resume_fname))
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        return classifier

    if args.arch == 'all':
        classifiers = {'alexnet': load_classifier('alexnet'),
                       'vgg16': load_classifier('vgg16'),
                       'resnet18': load_classifier('resnet18'), }
    else:
        classifiers = {args.arch: load_classifier(args.arch)}

    model = ColorCNN(args.backbone, args.temperature, args.bottleneck_channel, args.colors_channel,
                     args.topk, args.agg).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, 1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                 steps_per_epoch=len(train_loader), epochs=args.epochs)

    # first test the pre-trained classifier
    trainer = CNNTrainer(args, classifiers, logdir=logdir, sample_name='og_img')
    for arch in classifiers.keys():
        print(f'Test the pre-trained {arch}...')
        trainer.test(arch, test_loader, visualize=args.visualize)

    # then train ColorCNN
    trainer = CNNTrainer(args, classifiers, model, train_post_trans, logdir=logdir, pixsim_sample=pixsim_sample)

    def multicolor_test(arch, test_mode, epoch=None):
        if args.num_colors > 0:
            print(f'Testing {args.num_colors} colors on {arch}...')
            trainer.test(arch, test_loader, args.num_colors, epoch=epoch, visualize=args.visualize, test_mode=test_mode)
        else:
            for i in range(1, int(np.log2(-args.num_colors)) + 1):
                n_colors = 2 ** i
                test_set.num_colors[0] = n_colors
                print(f'Testing {n_colors} colors on {arch}...')
                trainer.test(arch, test_loader, n_colors, epoch=epoch, visualize=args.visualize, test_mode=test_mode)

    # learn
    if args.resume is None:
        print('Train ColorCNN...')
        for epoch in tqdm(range(1, args.epochs + 1)):
            print('Training...')
            trainer.train(epoch, train_loader, optimizer, args.num_colors, scheduler)
            if epoch % 20 == 0:
                for arch in classifiers.keys():
                    multicolor_test(arch, test_mode=args.mode, epoch=epoch)
            # save
            torch.save(model.state_dict(), os.path.join(logdir, 'ColorCNN.pth'))
    else:
        resume_fname = f'logs/colorcnn/{args.dataset}/{args.arch}/{args.num_colors}colors/{args.resume}/ColorCNN.pth'
        model.load_state_dict(torch.load(resume_fname))
    # test
    print(logdir)
    model.eval()
    print(f'Test in {args.mode} mode...')
    for arch in classifiers.keys():
        multicolor_test(arch, test_mode=args.mode, epoch=args.epochs)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='ColorCNN down sample')
    parser.add_argument('--num_colors', type=int, default=-64)
    parser.add_argument('--bottleneck_channel', type=int, default=16)
    parser.add_argument('--colors_channel', type=int, default=256)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--mode', type=str, default='classify', choices=['cluster', 'classify'])
    parser.add_argument('--cluster_loss', type=str, default='pixsim', choices=['pixsim', 'pixbce', 'ce', 'matchce'])
    parser.add_argument('--agg', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--ce_ratio', type=float, default=1.0, help='cross entropy loss')
    parser.add_argument('--kd_ratio', type=float, default=0.0, help='knowledge distillation loss')
    parser.add_argument('--perceptual_ratio', type=float, default=1.0, help='perceptual loss')
    parser.add_argument('--colormax_ratio', type=float, default=0.1, help='ensure all colors present')
    parser.add_argument('--pixsim_ratio', type=float, default=3.0, help='similarity towards the KMeans result')
    parser.add_argument('--recons_ratio', type=float, default=0.0, help='reconstruction loss')
    parser.add_argument('--conf_ratio', type=float, default=0.0,
                        help='softmax more like argmax (one-hot), reduce entropy of per-pixel color distribution')
    parser.add_argument('--info_ratio', type=float, default=0.1,
                        help='even distribution among all colors, increase entropy of entire-image color distribution')
    parser.add_argument('--color_jitter', type=float, default=1.0)
    parser.add_argument('--color_norm', type=float, default=4.0, help='normalizer for color palette')
    parser.add_argument('--color_dropout', type=float, default=0.0, help='dropout for color palette')
    parser.add_argument('--gaussian_noise', type=float, default=0.0, help='gaussian noise on quantized image')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for softmax')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'style14mini', 'imagenet', 'tiny200',
                                 'voc_cls', 'voc_seg'])
    parser.add_argument('-a', '--arch', type=str, default='all', choices=list(models.names()).extend('all'))
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--step_size', type=int, default=20, help='step_size for training')
    parser.add_argument('--task_update', type=int, default=20, help='task_update (num of batches) for training')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backbone', type=str, default='unet',
                        choices=['unet', 'unext', 'dncnn', 'cyclegan', 'styleunet', 'none'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    main(args)
