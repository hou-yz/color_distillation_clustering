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
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import color_distillation.utils.transforms as T
from color_distillation import models
from color_distillation.models.color_cnn import ColorCNN
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.load_checkpoint import checkpoint_loader
from color_distillation.utils.draw_curve import draw_curve
from color_distillation.utils.logger import Logger
from color_distillation.utils.image_utils import DeNormalize


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    data_path = os.path.expanduser('~/Data/') + args.dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    mean_var = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if args.dataset == 'svhn':
        H, W, C = 32, 32, 3
        num_class = 10

        train_trans = T.Compose([T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), ])

        train_set = datasets.SVHN(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.SVHN(data_path, split='test', download=True, transform=test_trans)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        H, W, C = 32, 32, 3
        num_class = 10 if args.dataset == 'cifar10' else 100

        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), ])

        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_trans)
        else:
            train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_trans)
    elif args.dataset == 'imagenet':
        H, W, C = 224, 224, 3
        num_class = 1000

        train_trans = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), ])

        train_set = datasets.ImageNet(data_path, split='train', transform=train_trans)
        test_set = datasets.ImageNet(data_path, split='val', transform=test_trans)
    elif args.dataset == 'stl10':
        H, W, C = 96, 96, 3
        num_class = 10
        # smaller batch size
        args.batch_size = 32

        train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), ])

        train_set = datasets.STL10(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=test_trans)
    elif args.dataset == 'tiny200':
        H, W, C = 64, 64, 3
        num_class = 200

        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.ToTensor(), ])

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    else:
        raise Exception

    # network specific setting
    if args.arch == 'alexnet':
        if 'cifar' not in args.dataset:
            args.color_norm = 1

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size * 2, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    logdir = f'logs/colorcnn/{args.dataset}/{args.arch}/{args.num_colors}colors/recons{args.recons_ratio}_' \
             f'colormax{args.colormax_ratio}_colorvar{args.colorvar_ratio}_conf{args.conf_ratio}_info{args.info_ratio}_' + \
             f'jitter{args.color_jitter}_colornorm{args.color_norm}_kd{args.kd_ratio}_perceptual{args.perceptual_ratio}_' \
             f'colordrop{args.color_dropout}_bottleneck{args.bottleneck_channel}_' + \
             datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./color_distillation', logdir + '/scripts/color_distillation')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    classifier = models.create(args.arch, num_class, not args.train_classifier).cuda()
    if not args.train_classifier:
        if args.dataset != 'imagenet':
            resume_fname = 'logs/grid/{}/{}/full_colors'.format(args.dataset, args.arch) + '/model.pth'
            classifier.load_state_dict(torch.load(resume_fname))
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False

    model = ColorCNN(args.backbone, args.temperature, args.color_norm, args.color_jitter, args.gaussian_noise,
                     args.color_dropout, args.bottleneck_channel).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, 1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                 steps_per_epoch=len(train_loader), epochs=args.epochs)

    # first test the pre-trained classifier
    print('Test the pre-trained classifier...')
    trainer = CNNTrainer(classifier, mean_var=mean_var, sample_name='og_img')
    trainer.test(test_loader, visualize=args.visualize)

    # then train ColorCNN
    trainer = CNNTrainer(classifier, model, mean_var=mean_var, label_smooth=args.label_smooth,
                         recons_ratio=args.recons_ratio,
                         kd_ratio=args.kd_ratio, perceptual_ratio=args.perceptual_ratio,
                         colormax_ratio=args.colormax_ratio, colorvar_ratio=args.colorvar_ratio,
                         conf_ratio=args.conf_ratio, info_ratio=args.info_ratio,
                         colormax_log_ratio=args.colormax_log_ratio)

    def test(test_mode='test_cluster'):
        if args.num_colors > 0:
            print(f'Testing {args.num_colors} colors...')
            trainer.test(test_loader, args.num_colors, visualize=args.visualize, test_mode=test_mode)
        else:
            for i in range(1, int(np.log2(-args.num_colors)) + 1):
                n_colors = 2 ** i
                print(f'Testing {n_colors} colors...')
                trainer.test(test_loader, n_colors, visualize=args.visualize, test_mode=test_mode)

    # learn
    if args.resume is None:
        print('Train ColorCNN...')
        for epoch in tqdm(range(1, args.epochs + 1)):
            print('Training...')
            trainer.train(epoch, train_loader, optimizer, args.num_colors, args.log_interval, scheduler)
            if epoch % 20 == 0:
                test(test_mode=args.mode)
        # save
        torch.save(model.state_dict(), os.path.join(logdir, 'ColorCNN.pth'))
    else:
        resume_dir = 'logs/colorcnn/{}/{}/{}colors/'.format(args.dataset, args.arch, args.num_colors) + args.resume
        resume_fname = resume_dir + '/ColorCNN.pth'
        model.load_state_dict(torch.load(resume_fname))
    # test
    model.eval()
    print(f'Test in {args.mode} mode...')
    test(test_mode=args.mode)
    # with adversarial
    if args.adversarial:
        print('********************    [adversarial first]    ********************')
        trainer = CNNTrainer(classifier, model, adversarial=args.adversarial, epsilon=args.epsilon, mean_var=mean_var,
                             sample_trans='colorcnn')
        print(f'Test in {args.mode} mode [adversarial: {args.adversarial} @ epsilon: {args.epsilon}]...')
        test(test_mode=args.mode)
        # print('********************    [quantization first]    ********************')
        # trainer = CNNTrainer(classifier, model, adversarial=args.adversarial, mean_var=mean_var)
        # print(f'Test in {args.mode} mode [adversarial: {args.adversarial}]...')
        test(test_mode=args.mode)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='ColorCNN down sample')
    parser.add_argument('--num_colors', type=int, default=-64)
    parser.add_argument('--bottleneck_channel', type=int, default=16)
    parser.add_argument('--mode', type=str, default='classify', choices=['cluster', 'classify'])
    parser.add_argument('--kd_ratio', type=float, default=0.0, help='knowledge distillation loss')
    parser.add_argument('--perceptual_ratio', type=float, default=0.0, help='perceptual loss')
    parser.add_argument('--colormax_ratio', type=float, default=1.0, help='ensure all colors present')
    parser.add_argument('--colormax_log_ratio', type=bool, default=0, help='use log ratio for colormax loss')
    parser.add_argument('--colorvar_ratio', type=float, default=0.0, help='color palette choose different colors')
    parser.add_argument('--recons_ratio', type=float, default=0.0, help='reconstruction loss')
    parser.add_argument('--conf_ratio', type=float, default=1.0,
                        help='softmax more like argmax (one-hot), reduce entropy of per-pixel color distribution')
    parser.add_argument('--info_ratio', type=float, default=1.0,
                        help='even distribution among all colors, increase entropy of entire-image color distribution')
    parser.add_argument('--color_jitter', type=float, default=1.0)
    parser.add_argument('--color_norm', type=float, default=4.0, help='normalizer for color palette')
    parser.add_argument('--color_dropout', type=float, default=0.0, help='dropout for color palette')
    parser.add_argument('--gaussian_noise', type=float, default=0.0, help='gaussian noise on quantized image')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature for softmax')
    parser.add_argument('--adversarial', default=None, type=str, choices=['fgsm', 'deepfool', 'bim', 'cw'])
    parser.add_argument('--epsilon', default=4, type=int)
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('-a', '--arch', type=str, default='vgg16', choices=models.names())
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--step_size', type=int, default=20, metavar='N', help='step_size for training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'dncnn', 'cyclegan'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train_classifier', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)
