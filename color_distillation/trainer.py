import os
import time
import numpy as np
import random
import color_distillation.utils.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image, make_grid
import foolbox as fb
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO
from sklearn import metrics
from color_distillation.evaluation.miou import eval_metrics
from color_distillation.loss import KDLoss, PixelSimLoss, ce_loss, matching_ce_loss
from color_distillation.models.alexnet import AlexNet
from color_distillation.models.vgg import VGG
from color_distillation.models.resnet import ResNet
from color_distillation.models.deeplabv3 import DeepLabV3
from color_distillation.utils.image_utils import Normalize


class CNNTrainer(object):
    def __init__(self, opts, classifiers, colorcnn=None, post_trans=None, logdir=None, pixsim_sample=False,
                 sample_name=None):
        self.opts = opts
        self.post_trans = post_trans
        self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        # network
        self.classifiers = classifiers
        self.colorcnn = colorcnn
        # loss
        self.CE_loss = nn.CrossEntropyLoss(label_smoothing=opts.label_smooth)
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.SEG_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.KD_loss = KDLoss()
        self.MSE_loss = nn.MSELoss()
        if opts.cluster_loss == 'pixsim':
            self.pixel_loss = PixelSimLoss(pixsim_sample)
        elif opts.cluster_loss == 'pixbce':
            self.pixel_loss = PixelSimLoss(pixsim_sample, normalize=False)
        elif opts.cluster_loss == 'ce':
            self.pixel_loss = lambda input, target: F.nll_loss(torch.log(input), target.argmax(dim=1))
        elif opts.cluster_loss == 'matchce':
            self.pixel_loss = matching_ce_loss
        else:
            self.pixel_loss = None
        # logging
        self.logdir = logdir
        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.sample_name = sample_name
        if colorcnn is not None:
            self.sample_name = 'colorcnn'

    def train(self, epoch, dataloader, optimizer, num_colors=None, cyclic_scheduler=None, ):
        if self.colorcnn:
            self.colorcnn.train()
            for classifier in self.classifiers.values():
                classifier.eval()
        else:
            for classifier in self.classifiers.values():
                classifier.train()

        def hook_for_arch(arch):
            def cnn_activation_hook(module, input, output):
                activations[arch].append(output)

            return cnn_activation_hook

        handles = {}
        for arch, classifier in self.classifiers.items():
            if isinstance(classifier, AlexNet) or isinstance(classifier, torchvision.models.AlexNet):
                handles[arch] = classifier.features[-2].register_forward_hook(hook_for_arch(arch))
            elif isinstance(classifier, VGG) or isinstance(classifier, torchvision.models.VGG):
                handles[arch] = classifier.features.register_forward_hook(hook_for_arch(arch))
            elif isinstance(classifier, ResNet) or isinstance(classifier, torchvision.models.ResNet):
                handles[arch] = classifier.layer4.register_forward_hook(hook_for_arch(arch))
            elif isinstance(classifier, DeepLabV3):
                handles[arch] = classifier.base.layer4.register_forward_hook(hook_for_arch(arch))
            else:
                raise Exception
        losses, correct, miss = 0, 0, 1e-8
        scores, labels = [], []
        t0 = time.time()

        # init
        if self.colorcnn:
            num_colors_batch = 2 ** np.random.randint(1, int(np.log2(-num_colors)) + 1) \
                if num_colors < 0 else num_colors
            dataloader.dataset.num_colors[0] = num_colors_batch
            # usable_ratio = 0

        for batch_idx, (img, target) in enumerate(dataloader):
            B, _, H, W = img.shape
            activations = {arch: [] for arch in self.classifiers}
            img = img.cuda()
            if isinstance(target, list):
                target, quantized_img, index_map = target[0].cuda(), target[1][0].cuda(), target[1][1].cuda()
                # mediancut_gt_mask = index_map.view([B, -1]).max(dim=1)[0] + 1 == num_colors_batch
                num_colors_batch = 2 ** int(np.log2(index_map.max().item() + 1) + 0.5)
            else:
                target = target.cuda()
            if self.colorcnn:
                # OG img
                if self.opts.kd_ratio or self.opts.perceptual_ratio:
                    with torch.no_grad():
                        output_targets = {arch: classifier(self.norm(img))
                                          for arch, classifier in self.classifiers.items()}
                    output_target = torch.cat(list(output_targets.values()))
                # colorcnn
                transformed_img, prob, color_palette = self.colorcnn(img, num_colors_batch, mode='train')
                # post process for augmentation
                if num_colors_batch <= 2 ** 3:
                    norm_color_palette = self.norm(color_palette.squeeze(4)).unsqueeze(4) / self.opts.color_norm
                    norm_color_palette = F.dropout3d(norm_color_palette.transpose(1, 2),
                                                     p=self.opts.color_dropout).transpose(1, 2)
                    jitter_color_palette = norm_color_palette + torch.randn(1).cuda() * \
                                           self.opts.color_jitter / np.log2(num_colors_batch)
                    norm_jit_img = (prob.unsqueeze(1) * jitter_color_palette).sum(dim=2)
                    norm_jit_img += self.opts.gaussian_noise * torch.randn_like(transformed_img)
                else:
                    norm_jit_img = self.norm(self.color_jitter(transformed_img))
                if self.opts.dataset == 'voc_seg':
                    trans_norm_jit_img, target = self.post_trans(norm_jit_img, target)
                else:
                    trans_norm_jit_img = self.post_trans(norm_jit_img)
                outputs = {arch: classifier(trans_norm_jit_img) for arch, classifier in self.classifiers.items()}
            else:
                outputs = {arch: classifier(self.norm(img)) for arch, classifier in self.classifiers.items()}
            # convert outputs dictionary into tensor
            output = torch.cat(list(outputs.values()))
            # repeat target accordingly
            target = target.repeat([len(self.classifiers)])
            if self.opts.dataset == 'voc_cls':
                ce_loss = self.BCE_loss(output, target)
                scores.append(torch.sigmoid(output).detach().cpu())
                labels.append(target.cpu())
            elif self.opts.dataset == 'voc_seg':
                B, H, W = target.shape
                output = F.interpolate(output, size=[H, W], mode='bilinear')
                ce_loss = self.SEG_loss(output, target)
                pred = torch.argmax(output, dim=1)
                correct += pred.eq(target).sum().item()
                miss += torch.tensor(target.shape).prod().item() - pred.eq(target).sum().item()
            else:
                ce_loss = self.CE_loss(output, target)
                pred = torch.argmax(output, 1)
                correct += pred.eq(target).sum().item()
                miss += target.shape[0] - pred.eq(target).sum().item()
            if self.colorcnn:
                entropy = lambda x: (-x * torch.log(x + 1e-16)).sum(dim=1)
                sigmoid = lambda x: 1 / (1 + np.exp(-x))

                B, _, H, W = img.shape
                # all colors taken
                color_appear_loss = -prob.view([B, -1, H * W]).max(dim=2)[0].mean()
                # per-pixel, higher confidence, reduce entropy of per-pixel color distribution
                conf_loss = entropy(prob).mean()
                # entire-image, even distribution among all colors, increase entropy of entire-image color distribution
                info_loss = -entropy(prob.mean(dim=[2, 3])).mean()
                recons_loss = self.MSE_loss(img, transformed_img)
                # print(f'batch{batch_idx}, num_colors_batch{num_colors_batch}, '
                #       f'max_index{torch.max(index_map.view(B, -1), dim=1)[0].float().mean().item() + 1}, '
                #       f'dataset.num_colors{dataloader.dataset.num_colors[0]}')
                ratio = sigmoid(np.log2(num_colors_batch) - 4)
                loss = self.opts.ce_ratio * ce_loss + \
                       self.opts.recons_ratio * recons_loss * np.log2(num_colors_batch) + \
                       self.opts.colormax_ratio * color_appear_loss + \
                       self.opts.conf_ratio * conf_loss + self.opts.info_ratio * info_loss
                if self.opts.kd_ratio or self.opts.perceptual_ratio:
                    # kd loss
                    kd_loss = self.KD_loss(output, output_target)
                    # perceptual loss
                    perceptual_loss = sum(self.MSE_loss(activations[arch][0], activations[arch][1])
                                          for arch in self.classifiers.keys()) / len(activations)
                    loss += self.opts.kd_ratio * kd_loss + self.opts.perceptual_ratio * perceptual_loss
                # usable_ratio += mediancut_gt_mask.float().mean()
                if self.opts.pixsim_ratio:
                    M = torch.zeros_like(prob).scatter(1, index_map, 1)
                    pixsim_loss = self.pixel_loss(prob, M)
                    loss += self.opts.pixsim_ratio * pixsim_loss * (ratio if self.opts.ce_ratio else 1)
            else:
                loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += ce_loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(dataloader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()

            if self.colorcnn and (num_colors < 0 and (batch_idx + 1) % self.opts.task_update == 0):
                dataloader.dataset.num_colors[0] = 2 ** np.random.randint(1, int(np.log2(-num_colors)) + 1)

            if (batch_idx + 1) % self.opts.log_interval == 0 or (batch_idx + 1) == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                if self.opts.dataset == 'voc_cls':
                    mAP = metrics.average_precision_score(torch.cat(labels, dim=0), torch.cat(scores, dim=0))
                    # accuracy = metrics.accuracy_score(torch.cat(labels, dim=0), torch.cat(scores, dim=0) > 0.5)
                    # hamming = metrics.hamming_loss(torch.cat(labels, dim=0), torch.cat(scores, dim=0) > 0.5)
                    print(f'Train epoch: {epoch}, batch:{batch_idx + 1}, \tloss: {losses / (batch_idx + 1):.3f}, '
                          f'mean average prec: {100. * mAP:.2f}%, time: {t_epoch:.3f}')
                else:
                    print(f'Train epoch: {epoch}, batch:{batch_idx + 1}, \tloss: {losses / (batch_idx + 1):.3f}, '
                          f'prec: {100. * correct / (correct + miss):.1f}%, time: {t_epoch:.3f}')
                if self.colorcnn:
                    log = f'ce: {ce_loss.item():.3f}, recons: {recons_loss.item():.3f}, colormax: ' \
                          f'{color_appear_loss.item():.3f}, conf: {conf_loss.item():.3f}, info: {info_loss.item():.3f}'
                    if self.opts.pixsim_ratio:
                        log += f', pixsim: {pixsim_loss.item():.3f}'
                    if self.opts.kd_ratio or self.opts.perceptual_ratio:
                        log += f', kd: {kd_loss.item():.3f}, prcp: {kd_loss.item():.3f}'
                    # log += f', usable_ratio: {usable_ratio / (batch_idx + 1):.3f}'
                    print(log)

                if self.logdir is not None and self.colorcnn:
                    self.sample_image(img, num_colors_batch, quantized_img=quantized_img,
                                      fname=f'{self.logdir}/imgs/{epoch:03d}_train.png')

        for arch, handle in handles.items():
            handle.remove()

        return losses / len(dataloader), correct / (correct + miss)

    def test(self, arch, dataloader, num_colors=None, epoch=None, visualize=False, test_mode='test'):
        classifier = self.classifiers[arch]
        classifier.eval()

        activation = {}
        recons_loss_list = []
        os.makedirs(f'{self.logdir}/imgs', exist_ok=True)

        def classifier_activation_hook(module, input, output):
            activation['classifier'] = output.cpu().detach().numpy()

        def auto_encoder_activation_hook(module, input, output):
            activation['auto_encoder'] = output.cpu().detach().numpy()

        def visualize_img(i):
            # index = [8, 49, 50, 61]
            # index = [28 + 64, 40 + 64, 59 + 64, 128, 136]
            # index = [28, 31, 40, 43, 5, 6]
            # index = target.sum(dim=1) > 1
            image_grid = make_grid(data_quantized[[2, 10, 13, 15, 20]], nrow=8, normalize=False)
            # Arange images along y-axis
            save_image(image_grid, f'{self.logdir}/imgs/batch_{num_colors}.png', normalize=False)

            if self.colorcnn:
                og_img = img[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(og_img)
                plt.show()
                og_img = Image.fromarray((og_img * 255).astype('uint8')).resize((512, 512), Image.NEAREST)
                og_img.save(f'{self.logdir}/imgs/og_img.png')

                downsampled_img = data_quantized[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(downsampled_img)
                plt.show()
                downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8')).resize((512, 512),
                                                                                                  Image.NEAREST)
                downsampled_img.save(f'{self.logdir}/imgs/colorcnn_{num_colors}.png')

                fig = plt.figure(frameon=False)
                fig.set_size_inches(2, 2)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(np.linalg.norm(activation['auto_encoder'][i], axis=0), cmap='viridis')
                plt.savefig(f'{self.logdir}/imgs/encoder_{num_colors}.png')
                plt.show()
                # index map
                plt.imshow(M[i, 0].cpu().numpy(), cmap='Blues')
                plt.savefig(f'{self.logdir}/imgs/M_{num_colors}.png', bbox_inches='tight')
                plt.show()
            else:
                downsampled_img = data_quantized[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(downsampled_img)
                plt.show()
                downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8')).resize((512, 512),
                                                                                                  Image.NEAREST)
                downsampled_img.save(f'{self.logdir}/imgs/{self.sample_name}_{num_colors}.png')
            if 'voc' not in self.opts.dataset:
                cam_map = np.sum(activation['classifier'][i] * weight_softmax[pred[i].item()].reshape((-1, 1, 1)),
                                 axis=0)
                cam_map = cam_map - np.min(cam_map)
                cam_map = cam_map / np.max(cam_map)
                cam_map = np.uint8(255 * cam_map)
                cam_map = cv2.resize(cam_map, (downsampled_img.size))
                heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
                downsampled_img = cv2.cvtColor(np.asarray(downsampled_img), cv2.COLOR_RGB2BGR)
                cam_result = np.uint8(heatmap * 0.3 + downsampled_img * 0.5)
                # cam_result = cv2.putText(cam_result, '{:.1f}%, {}'.format(
                #     100 * F.softmax(output, dim=1)[i, target[i]].item(),
                #     'Success' if pred.eq(target)[i].item() else 'Failure'),
                #                          (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                #                          (0, 255, 0) if pred.eq(target)[i].item() else (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(f'{self.logdir}/imgs/{self.sample_name}_cam_{num_colors}.png', cam_result)
                plt.imshow(cv2.cvtColor(cam_result, cv2.COLOR_BGR2RGB))
                plt.show()
                # ax.imshow(cam_map, cmap='viridis', aspect='auto')
                # fig.savefig("activation.png", )
                # fig.show()

        buffer_size_counter, number_of_colors, dataset_size = 0, 0, 1e-8
        if self.colorcnn:
            self.colorcnn.eval()
        losses, correct, miss = 0, 0, 1e-8
        scores, labels = [], []
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        H = W = 1
        t0 = time.time()

        if visualize:
            if hasattr(classifier, 'features'):
                classifier_handle = classifier.features.register_forward_hook(classifier_activation_hook)
            else:
                classifier_handle = classifier.layer4.register_forward_hook(classifier_activation_hook)
            if hasattr(classifier, 'classifier'):
                classifier_layer = classifier.classifier
            else:
                classifier_layer = classifier.fc
            if isinstance(classifier_layer, nn.Sequential):
                classifier_layer = classifier_layer[-1]
            weight_softmax = classifier_layer.weight.detach().cpu().numpy()
            if self.colorcnn:
                colorcnn_handle = self.colorcnn.base.register_forward_hook(auto_encoder_activation_hook)

        for batch_idx, (img, target) in enumerate(dataloader):
            img = img.cuda()
            B, C, H, W = img.shape

            if isinstance(target, list):
                target, quantized_img, index_map = target[0].cuda(), target[1][0].cuda(), target[1][1].cuda()
            else:
                target = target.cuda()

            if self.colorcnn:
                with torch.no_grad():
                    data_quantized, prob, _ = self.colorcnn(img, num_colors, mode=test_mode)
                    data_quantized = data_quantized.clamp(0, 1)
            else:
                data_quantized = img

            recons_loss_list.append(self.MSE_loss(img, data_quantized).item())

            with torch.no_grad():
                output = classifier(self.norm(data_quantized))
            if self.opts.dataset == 'voc_cls':
                loss = self.BCE_loss(output, target)
                scores.append(torch.sigmoid(output).cpu())
                labels.append(target.cpu())
            elif self.opts.dataset == 'voc_seg':
                B, H, W = target.shape
                output = F.interpolate(output, size=[H, W], mode='bilinear')
                loss = self.SEG_loss(output, target)
                correct, labeled, inter, union = eval_metrics(output, target, 21, 255)

                # PRINT INFO
                total_inter, total_union = total_inter + inter, total_union + union
                total_correct, total_label = total_correct + correct, total_label + labeled
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean()
                seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                               # "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
                               }
            else:
                pred = torch.argmax(output, 1)
                correct += pred.eq(target).sum().item()
                miss += target.shape[0] - pred.eq(target).sum().item()
                loss = self.CE_loss(output, target)
            losses += loss.item()
            # image file size
            if self.colorcnn:
                M = torch.argmax(prob, dim=1, keepdim=True)  # argmax color index map
                for i in range(target.shape[0]):
                    number_of_colors += len(M[0].unique())
                    downsampled_img = data_quantized[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                    downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8'))

                    png_buffer = BytesIO()
                    downsampled_img.save(png_buffer, "PNG")
                    buffer_size_counter += png_buffer.getbuffer().nbytes
                    dataset_size += 1
            # plotting
            if visualize:
                self.sample_image(img.cuda(), num_colors,
                                  fname=f'{self.logdir}/imgs/{self.opts.epochs:03d}_{num_colors}_{batch_idx}.png',
                                  quantized_img=quantized_img)
                if batch_idx == 3:
                    i = 105
                    visualize_img(i)
                    if 'voc' not in self.opts.dataset:
                        print(f'pred: {pred[i]}, target: {target[i]}, '
                              f'conf: {torch.softmax(output, 1)[i][target[i]] * 100:.1f}%')
                    break

        if self.opts.dataset == 'voc_cls':
            labels, scores = torch.cat(labels, dim=0), torch.cat(scores, dim=0)
            mAP = metrics.average_precision_score(labels, scores)
            accuracy = metrics.accuracy_score(labels, scores > 0.5)
            # f1 = metrics.f1_score(labels, scores > 0.5, average='macro')
            hamming = metrics.hamming_loss(labels, scores > 0.5)
            print(f'Test, loss: {losses / len(dataloader):.3f}, mean average prec: {100. * mAP:.2f}%, '
                  f'accuracy: {100. * accuracy:.2f}%, hamming loss: {hamming:.4f}, ')
        elif self.opts.dataset == 'voc_seg':
            print(f'Test, loss: {losses / len(dataloader):.3f}, {seg_metrics}, ')
        else:
            print(f'Test, loss: {losses / len(dataloader):.3f}, prec: {100. * correct / (correct + miss):.2f}%, ')
        print(f'time: {time.time() - t0:.1f}, recons loss: {np.mean(recons_loss_list):.3f}')
        if self.colorcnn:
            print(f'Average number of colors per image: {number_of_colors / dataset_size:.1f}; \n'
                  f'Average image size: {buffer_size_counter / dataset_size:.1f}; '
                  f'Bit per pixel: {buffer_size_counter / dataset_size / H / W:.3f}')

        if visualize:
            classifier_handle.remove()
            if self.colorcnn:
                colorcnn_handle.remove()

        if epoch is not None and self.logdir is not None:
            it = iter(dataloader)
            for i in range(4):
                img, tgt = next(it)
                quantized_img = tgt[1][0].cuda() if isinstance(tgt, list) else None
                self.sample_image(img.cuda(), num_colors, fname=f'{self.logdir}/imgs/{epoch:03d}_{num_colors}_{i}.png',
                                  quantized_img=quantized_img)

        if self.opts.dataset == 'voc_cls':
            return losses / len(dataloader), mAP
        elif self.opts.dataset == 'voc_seg':
            return losses / len(dataloader), mIoU
        else:
            return losses / len(dataloader), correct / (correct + miss)

    def sample_image(self, img, num_colors, fname, quantized_img=None, nrow=4):
        if self.colorcnn and num_colors is not None:
            """Saves a generated sample from the test set"""
            self.colorcnn.eval()
            with torch.no_grad():
                img_trans, _, _ = self.colorcnn(img, num_colors, mode='test')
            OG = make_grid(img, nrow=nrow)
            colorcnn = make_grid(img_trans, nrow=nrow)
            if quantized_img is not None:
                mediancut = make_grid(quantized_img, nrow=nrow)
                res = [OG, colorcnn, mediancut]
            else:
                res = [OG, colorcnn]
            # Arange images along x-axis
            image_grid = torch.cat(res, 2)
        else:
            image_grid = make_grid(img, nrow=nrow)
        save_image(image_grid, fname, normalize=False)
