import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from color_distillation import datasets
import color_distillation.utils.transforms as T
import color_distillation.utils.ext_transforms as exT
from color_distillation import models
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.sampler import RandomSeqSampler
from color_distillation.utils.draw_curve import draw_curve
from color_distillation.utils.logger import Logger
from color_distillation.utils.buffer_size_counter import BufferSizeCounter


def main(args):
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

    buffer_size_counter = BufferSizeCounter()
    if args.sample_name == 'mcut':
        sample_trans = [T.MedianCut(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'octree':
        sample_trans = [T.OCTree(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'kmeans':
        sample_trans = [T.KMeans(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'jpeg':
        sample_trans = [T.JpegCompression(buffer_size_counter, args.jpeg_ratio)]
    elif args.sample_name == 'canny':
        sample_trans = [T.CannyEdge(), T.PNGCompression(buffer_size_counter)]
    elif args.sample_name is None:
        sample_trans = [T.PNGCompression(buffer_size_counter)]
        args.sample_name = 'og_img'
    else:
        raise Exception

    args.cluster_loss = None

    # dataset
    data_path = os.path.expanduser('~/Data/') + args.dataset
    base_lr_ratio = 1
    pretrained = False
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        num_class = 10 if args.dataset == 'cifar10' else 100

        og_train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        if args.dataset == 'cifar10':
            og_train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=og_train_trans)
            og_test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=og_test_trans)
            sampled_test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=sampled_test_trans)
        else:
            og_train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=og_train_trans)
            og_test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=og_test_trans)
            sampled_test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=sampled_test_trans)
    elif args.dataset == 'imagenet':
        num_class = 1000
        args.batch_size = 32
        pretrained = True

        og_train_trans = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.Resize(256), T.CenterCrop(224), T.ToTensor(), ])

        og_train_set = datasets.ImageNet(data_path, split='train', transform=og_train_trans)
        og_test_set = datasets.ImageNet(data_path, split='val', transform=og_test_trans)
        sampled_test_set = datasets.ImageNet(data_path, split='val', transform=sampled_test_trans)
    elif args.dataset == 'style14mini':
        num_class = 14
        args.lr = 0.01
        args.batch_size = 32
        # base_lr_ratio = 0.1
        # pretrained = True

        og_train_trans = T.Compose([T.RandomResizedCrop(112), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.Resize(128), T.CenterCrop(112), T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.Resize(128), T.CenterCrop(112), T.ToTensor(), ])

        og_train_set = datasets.ImageFolder(data_path + '/train', transform=og_train_trans)
        og_test_set = datasets.ImageFolder(data_path + '/val', transform=og_test_trans)
        sampled_test_set = datasets.ImageFolder(data_path + '/val', transform=sampled_test_trans)
    elif args.dataset == 'stl10':
        num_class = 10
        # smaller batch size
        args.batch_size = 32

        og_train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        og_train_set = datasets.STL10(data_path, split='train', download=True, transform=og_train_trans)
        og_test_set = datasets.STL10(data_path, split='test', download=True, transform=og_test_trans)
        sampled_test_set = datasets.STL10(data_path, split='test', download=True, transform=sampled_test_trans)
    elif args.dataset == 'tiny200':
        num_class = 200

        og_train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        og_train_set = datasets.ImageFolder(data_path + '/train', transform=og_train_trans)
        og_test_set = datasets.ImageFolder(data_path + '/val', transform=og_test_trans)
        sampled_test_set = datasets.ImageFolder(data_path + '/val', transform=sampled_test_trans)
    elif args.dataset == 'voc_cls':
        num_class = 20
        args.batch_size = 32
        args.log_interval = 200
        # base_lr_ratio = 0.1
        # pretrained = True
        data_path = os.path.expanduser('~/Data/pascal_VOC')

        og_train_trans = T.Compose([T.RandomResizedCrop(112), T.RandomHorizontalFlip(), T.ToTensor(), ])
        og_test_trans = T.Compose([T.Resize(128), T.CenterCrop(112), T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.Resize(128), T.CenterCrop(112), T.ToTensor(), ])

        og_train_set = datasets.VOCClassification(data_path, image_set='train', transform=og_train_trans)
        og_test_set = datasets.VOCClassification(data_path, image_set='val', transform=og_test_trans)
        sampled_test_set = datasets.VOCClassification(data_path, image_set='val', transform=sampled_test_trans)
    elif args.dataset == 'voc_seg':
        num_class = 21
        args.lr = 0.01
        args.batch_size = 8
        args.arch = 'deeplab'
        args.log_interval = 2000
        base_lr_ratio = 0.1
        crop_size = [160, 160]
        data_path = os.path.expanduser('~/Data/pascal_VOC')

        og_train_trans = exT.ExtCompose([exT.ExtResize(crop_size), exT.ExtRandomScale(scale=(0.5, 2)),
                                         exT.ExtRandomCrop(size=crop_size, pad_if_needed=True),
                                         exT.ExtRandomHorizontalFlip(), exT.ExtToTensor()])
        og_test_trans = exT.ExtCompose([exT.ExtResize(crop_size), exT.ExtToTensor(), ])
        sampled_test_trans = exT.ExtCompose(sample_trans + [exT.ExtResize(crop_size), exT.ExtToTensor(), ])

        og_train_set = datasets.VOCSegmentation(data_path, image_set='train', transforms=og_train_trans)
        og_test_set = datasets.VOCSegmentation(data_path, image_set='val', transforms=og_test_trans)
        sampled_test_set = datasets.VOCSegmentation(data_path, image_set='val', transforms=sampled_test_trans)
    else:
        raise Exception

    og_train_loader = DataLoader(og_train_set, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)
    og_test_loader = DataLoader(og_test_set, batch_size=args.batch_size,
                                # sampler=RandomSeqSampler(og_test_set),
                                num_workers=args.num_workers, pin_memory=True)
    sampled_test_loader = DataLoader(sampled_test_set, batch_size=args.batch_size,
                                     # sampler=RandomSeqSampler(og_test_set),
                                     num_workers=args.num_workers, pin_memory=True)

    logdir = 'logs/grid/{}/{}/{}colors'.format(args.dataset, args.arch,
                                               'full_')  # if args.num_colors is None else args.num_colors
    if args.train:
        os.makedirs(logdir, exist_ok=True)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    model = models.create(args.arch, num_class, pretrained).cuda()
    if args.dataset == 'voc_seg':
        param_dicts = [{"params": [p for n, p in model.named_parameters() if 'base' not in n and p.requires_grad], },
                       {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                        "lr": args.lr * base_lr_ratio, }, ]
        optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

        def lr_scheduler_wrapper(epochs, len_loader, decay=0.9, warmup=0):
            def warm_and_decay_lr_scheduler(step: int):
                total_steps = epochs * len_loader
                warmup_steps = warmup * total_steps
                assert step <= total_steps
                if step < warmup_steps:
                    factor = step / warmup_steps
                else:
                    factor = (1 - step / (total_steps + 1e-8)) ** decay
                return factor

            return warm_and_decay_lr_scheduler

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lr_scheduler_wrapper(args.epochs, len(og_train_loader)))

    else:
        param_dicts = [{"params": [p for n, p in model.named_parameters() if
                                   'classifier' not in n and 'fc' not in n and p.requires_grad],
                        "lr": args.lr * base_lr_ratio},
                       {"params": [p for n, p in model.named_parameters() if
                                   ('classifier' in n or 'fc' in n) and p.requires_grad], }]
        optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                        steps_per_epoch=len(og_train_loader), epochs=args.epochs)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    trainer = CNNTrainer(args, {args.arch: model}, logdir=logdir, sample_name=args.sample_name)

    # learn
    if args.train:
        for epoch in tqdm(range(1, args.epochs + 1)):
            print('Train on original dateset...')
            train_loss, train_prec = trainer.train(epoch, og_train_loader, optimizer, cyclic_scheduler=scheduler)
            # scheduler.step()
            print('Test on original dateset...')
            og_test_loss, og_test_prec = trainer.test(args.arch, og_test_loader)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            og_test_loss_s.append(og_test_loss)
            og_test_prec_s.append(og_test_prec)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       og_test_loss_s, og_test_prec_s, )
        # save
        torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))
    else:
        if args.dataset != 'imagenet':
            model.load_state_dict(torch.load(logdir + '/model.pth'))
    pass
    # test
    model.eval()
    print('Test on sampled dateset...')
    img, tgt = next(iter(og_test_loader))
    B, C, H, W = img.shape
    trainer.test(args.arch, sampled_test_loader, args.num_colors, args.epochs, visualize=args.visualize)
    print(f'Average image size: {buffer_size_counter.size[0] / len(sampled_test_set):.1f}; '
          f'Bit per pixel: {buffer_size_counter.size[0] / len(sampled_test_set) / H / W:.3f}')
    pass


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Grid-wise down sample')
    parser.add_argument('--num_colors', type=int, default=None, help='down sample ratio for area')
    parser.add_argument('--sample_name', type=str, default=None,
                        choices=['mcut', 'octree', 'kmeans', 'jpeg', 'canny'])
    parser.add_argument('--dither', action='store_true', default=False)
    parser.add_argument('--jpeg_ratio', type=int, default=50)
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'style14mini', 'imagenet', 'tiny200',
                                 'voc_cls', 'voc_seg'])
    parser.add_argument('-a', '--arch', type=str, default='vgg16', choices=models.names())
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)
