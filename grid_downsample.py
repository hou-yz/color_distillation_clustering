import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import color_distillation.utils.transforms as T
from color_distillation import models
from color_distillation.trainer import CNNTrainer
from color_distillation.utils.load_checkpoint import checkpoint_loader
from color_distillation.utils.draw_curve import draw_curve
from color_distillation.utils.logger import Logger
from color_distillation.utils.buffer_size_counter import BufferSizeCounter


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if args.dataset == 'svhn' or args.dataset == 'cifar10' or args.dataset == 'cifar100':
        H, W, C = 32, 32, 3
    elif args.dataset == 'imagenet':
        H, W, C = 224, 224, 3
    elif args.dataset == 'stl10':
        H, W, C = 96, 96, 3
    elif args.dataset == 'tiny200':
        H, W, C = 64, 64, 3
    else:
        raise Exception

    buffer_size_counter = BufferSizeCounter()
    if args.sample_name == 'mcut':
        sample_trans = [T.MedianCut(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        test_sample_trans = T.MedianCut(args.num_colors, args.dither)
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'octree':
        sample_trans = [T.OCTree(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        test_sample_trans = T.OCTree(args.num_colors, args.dither)
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'kmeans':
        sample_trans = [T.KMeans(args.num_colors, args.dither), T.PNGCompression(buffer_size_counter)]
        test_sample_trans = T.KMeans(args.num_colors, args.dither)
        if args.dither: args.sample_name += '_dither'
    elif args.sample_name == 'jpeg':
        sample_trans = [T.JpegCompression(buffer_size_counter, args.jpeg_ratio)]
        test_sample_trans = T.JpegCompression(buffer_size_counter, args.jpeg_ratio)
    elif args.sample_name is None:
        sample_trans = [T.PNGCompression(buffer_size_counter)]
        test_sample_trans = T.PNGCompression(buffer_size_counter)
        args.sample_name = 'og_img'
    else:
        raise Exception

    # dataset
    data_path = os.path.expanduser('~/Data/') + args.dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    mean_var = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_sample_trans = T.Compose([T.ToPILImage(), test_sample_trans, T.ToTensor()])
    if args.dataset == 'svhn':
        num_class = 10

        sampled_train_trans = T.Compose(sample_trans + [T.ToTensor(), normalize, ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        sampled_train_set = datasets.SVHN(data_path, split='train', download=True, transform=sampled_train_trans)
        og_test_set = datasets.SVHN(data_path, split='test', download=True, transform=og_test_trans)
        sampled_test_set = datasets.SVHN(data_path, split='test', download=True, transform=sampled_test_trans)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar100':
        num_class = 10 if args.dataset == 'cifar10' else 100

        sampled_train_trans = T.Compose(sample_trans + [T.RandomCrop(32, padding=4),
                                                        T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        if args.dataset == 'cifar10':
            sampled_train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=sampled_train_trans)
            og_test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=og_test_trans)
            sampled_test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=sampled_test_trans)
        else:
            sampled_train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=sampled_train_trans)
            og_test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=og_test_trans)
            sampled_test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=sampled_test_trans)
    elif args.dataset == 'imagenet':
        num_class = 1000

        sampled_train_trans = T.Compose(sample_trans + [T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                                                        T.ToTensor(), normalize, ])
        og_test_trans = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.Resize(256), T.CenterCrop(224), T.ToTensor(), ])

        sampled_train_set = datasets.ImageNet(data_path, split='train', transform=sampled_train_trans, )
        og_test_set = datasets.ImageNet(data_path, split='val', transform=og_test_trans)
        sampled_test_set = datasets.ImageNet(data_path, split='val', transform=sampled_test_trans, )
    elif args.dataset == 'stl10':
        num_class = 10
        # smaller batch size
        args.batch_size = 32

        sampled_train_trans = T.Compose(sample_trans + [T.RandomCrop(96, padding=12),
                                                        T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        sampled_train_set = datasets.STL10(data_path, split='train', download=True, transform=sampled_train_trans)
        og_test_set = datasets.STL10(data_path, split='test', download=True, transform=og_test_trans)
        sampled_test_set = datasets.STL10(data_path, split='test', download=True, transform=sampled_test_trans)
    elif args.dataset == 'tiny200':
        num_class = 200

        sampled_train_trans = T.Compose(sample_trans + [T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),
                                                        T.ToTensor(), normalize, ])
        og_test_trans = T.Compose([T.ToTensor(), ])
        sampled_test_trans = T.Compose(sample_trans + [T.ToTensor(), ])

        sampled_train_set = datasets.ImageFolder(data_path + '/train', transform=sampled_train_trans, )
        og_test_set = datasets.ImageFolder(data_path + '/val', transform=og_test_trans)
        sampled_test_set = datasets.ImageFolder(data_path + '/val', transform=sampled_test_trans, )
    else:
        raise Exception

    sampled_train_loader = torch.utils.data.DataLoader(sampled_train_set, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_workers, pin_memory=True)
    og_test_loader = torch.utils.data.DataLoader(og_test_set, batch_size=args.batch_size * 2, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
    sampled_test_loader = torch.utils.data.DataLoader(sampled_test_set, batch_size=args.batch_size * 2, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)

    logdir = 'logs/grid/{}/{}/{}colors'.format(args.dataset, args.arch,
                                               'full_' if args.num_colors is None else args.num_colors)
    if args.train:
        os.makedirs(logdir, exist_ok=True)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    model = models.create(args.arch, num_class, not args.train).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    steps_per_epoch=len(sampled_train_loader), epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, 1)

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    masked_test_loss_s = []
    masked_test_prec_s = []

    trainer = CNNTrainer(args, model, mean_var=mean_var, sample_name=args.sample_name)

    # learn
    if args.train:
        for epoch in tqdm(range(1, args.epochs + 1)):
            print('Train on sampled dateset...')
            train_loss, train_prec = trainer.train(epoch, sampled_train_loader, optimizer,
                                                   log_interval=args.log_interval, cyclic_scheduler=scheduler)
            print('Test on original dateset...')
            og_test_loss, og_test_prec = trainer.test(og_test_loader)
            print('Test on sampled dateset...')
            masked_test_loss, masked_test_prec = trainer.test(sampled_test_loader)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            og_test_loss_s.append(og_test_loss)
            og_test_prec_s.append(og_test_prec)
            masked_test_loss_s.append(masked_test_loss)
            masked_test_prec_s.append(masked_test_prec)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       og_test_loss_s, og_test_prec_s, masked_test_loss_s, masked_test_prec_s)
        # save
        torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))
    else:
        if args.dataset != 'imagenet':
            resume_dir = 'logs/grid/{}/{}/full_colors'.format(args.dataset, args.arch)
            resume_fname = resume_dir + '/model.pth'
            model.load_state_dict(torch.load(resume_fname))
    pass
    # test
    model.eval()
    # without adversarial
    print('Test on sampled dateset...')
    trainer.test(sampled_test_loader, args.num_colors, visualize=args.visualize)
    print(f'Average image size: {buffer_size_counter.size / len(sampled_test_set):.1f}; '
          f'Bit per pixel: {buffer_size_counter.size / len(sampled_test_set) / H / W:.3f}')
    buffer_size_counter.reset()
    # with adversarial
    if args.adversarial:
        print('********************    [adversarial first]    ********************')
        trainer = CNNTrainer(args, model, adversarial=args.adversarial, epsilon=args.epsilon,
                             mean_var=mean_var, sample_name=args.sample_name, sample_trans=test_sample_trans)
        print(f'Test on sampled dateset [adversarial: {args.adversarial} @ epsilon: {args.epsilon}]...')
        trainer.test(og_test_loader, args.num_colors, visualize=args.visualize)
        print(f'Average image size: {buffer_size_counter.size / len(sampled_test_set):.1f}; '
              f'Bit per pixel: {buffer_size_counter.size / len(sampled_test_set) / H / W:.3f}')
        buffer_size_counter.reset()

        # print('********************    [quantization first]    ********************')
        # trainer = CNNTrainer(model, adversarial=args.adversarial,
        #                      mean_var=mean_var, sample_method=args.sample_name)
        # print(f'Test on sampled dateset [adversarial: {args.adversarial}]...')
        # trainer.test(sampled_test_loader, visualize=args.visualize)
        # print(f'Average image size: {buffer_size_counter.size / len(sampled_test_set):.1f}; '
        #       f'Bit per pixel: {buffer_size_counter.size / len(sampled_test_set) / H / W:.3f}')
    pass


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Grid-wise down sample')
    parser.add_argument('--num_colors', type=int, default=None, help='down sample ratio for area')
    parser.add_argument('--sample_name', type=str, default=None,
                        choices=['mcut', 'octree', 'kmeans', 'jpeg'])
    parser.add_argument('--dither', action='store_true', default=False)
    parser.add_argument('--jpeg_ratio', type=int, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--adversarial', default=None, type=str, choices=['fgsm', 'deepfool', 'bim', 'cw'])
    parser.add_argument('--epsilon', default=4, type=int)
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('-a', '--arch', type=str, default='vgg16', choices=models.names())
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--step_size', type=int, default=20, metavar='N', help='step_size for training (default: 10)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    args = parser.parse_args()

    main(args)
