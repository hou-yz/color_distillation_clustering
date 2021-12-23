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
from color_distillation.loss import LabelSmoothLoss, KDLoss, PixelSimLoss
from color_distillation.models.alexnet import AlexNet
from color_distillation.models.vgg import VGG
from color_distillation.models.resnet import ResNet
from color_distillation.utils.image_utils import Normalize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CNNTrainer(object):
    def __init__(self, opts, classifier, colorcnn=None, post_trans=None, logdir=None, pixsim_sample=False,
                 adversarial=None, epsilon=2, sample_name=None, sample_trans=None):
        self.opts = opts
        self.post_trans = post_trans
        self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        # network
        self.classifier = classifier
        self.colorcnn = colorcnn
        # loss
        self.CE_loss = LabelSmoothLoss(opts.label_smooth) if opts.label_smooth else nn.CrossEntropyLoss()
        self.KD_loss = KDLoss()
        self.MSE_loss = nn.MSELoss()
        self.pixel_loss = PixelSimLoss(pixsim_sample)
        # logging
        self.logdir = logdir
        self.norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # adversarial
        if adversarial:
            self.fclassifier = fb.PyTorchModel(self.classifier, bounds=(0, 1),
                                               preprocessing=dict(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225], axis=-3))
        if adversarial == 'fgsm':
            self.adversarial = fb.attacks.LinfFastGradientAttack()
        elif adversarial == 'deepfool':
            self.adversarial = fb.attacks.LinfDeepFoolAttack()
        elif adversarial == 'bim':
            self.adversarial = fb.attacks.LinfBasicIterativeAttack()
        elif adversarial == 'cw':
            self.adversarial = fb.attacks.L2CarliniWagnerAttack(steps=10)
        else:
            self.adversarial = None
        self.epsilon = epsilon / 255
        self.sample_name = sample_name
        if colorcnn is not None:
            self.sample_name = 'colorcnn'
        self.sample_trans = T.Compose([T.ToPILImage()] + sample_trans + [T.ToTensor()]) \
            if isinstance(sample_trans, list) else sample_trans

    def train(self, epoch, dataloader, optimizer, num_colors=None, cyclic_scheduler=None, ):
        if self.colorcnn:
            self.colorcnn.train()
            self.classifier.eval()
        else:
            self.classifier.train()

        def cnn_activation_hook(module, input, output):
            activation.append(output)

        if isinstance(self.classifier, AlexNet) or isinstance(self.classifier, torchvision.models.AlexNet):
            handle = self.classifier.features[-2].register_forward_hook(cnn_activation_hook)
        elif isinstance(self.classifier, VGG) or isinstance(self.classifier, torchvision.models.VGG):
            handle = self.classifier.features.register_forward_hook(cnn_activation_hook)
        elif isinstance(self.classifier, ResNet) or isinstance(self.classifier, torchvision.models.ResNet):
            handle = self.classifier.layer4.register_forward_hook(cnn_activation_hook)
        else:
            raise Exception
        losses, correct, miss = 0, 0, 0
        t0 = time.time()

        # init
        if self.colorcnn and num_colors:
            num_colors_batch = 2 ** np.random.randint(1, int(np.log2(-num_colors)) + 1) \
                if num_colors < 0 else num_colors
            dataloader.dataset.num_colors[0] = num_colors_batch

        for batch_idx, (img, target) in enumerate(dataloader):
            activation = []
            img = img.cuda()
            if isinstance(target, list):
                label, quantized_img, index_map = target[0].cuda(), target[1][0].cuda(), target[1][1].cuda()
                num_colors_batch = 2 ** int(np.log2(index_map.max().item() + 1) + 0.5)
            else:
                label = target.cuda()
            if self.colorcnn:
                # OG img
                if self.opts.kd_ratio or self.opts.perceptual_ratio:
                    output_target = self.classifier(self.norm(img))
                # colorcnn
                transformed_img, prob, color_palette = self.colorcnn(img, num_colors_batch, mode='train')
                # post process for augmentation
                if num_colors_batch <= 2 ** 3:
                    color_palette = color_palette.clamp(0, 1)
                    norm_color_palette = self.norm(color_palette.squeeze(4)).unsqueeze(4) / self.opts.color_norm
                    norm_color_palette = F.dropout3d(norm_color_palette.transpose(1, 2),
                                                     p=self.opts.color_dropout).transpose(1, 2)
                    jitter_color_palette = norm_color_palette + torch.randn(1).cuda() * \
                                           self.opts.color_jitter / np.log2(num_colors_batch)
                    norm_jit_img = (prob.unsqueeze(1) * jitter_color_palette).sum(dim=2)
                    norm_jit_img += self.opts.gaussian_noise * torch.randn_like(transformed_img)
                else:
                    transformed_img = transformed_img.clamp(0, 1)
                    norm_jit_img = self.norm(self.color_jitter(transformed_img))
                trans_norm_jit_img = self.post_trans(norm_jit_img)
                output = self.classifier(trans_norm_jit_img)
            else:
                output = self.classifier(self.norm(img))
            pred = torch.argmax(output, 1)
            correct += pred.eq(label).sum().item()
            miss += label.shape[0] - pred.eq(label).sum().item()
            ce_loss = self.CE_loss(output, label)
            if self.colorcnn:
                B, _, H, W = img.shape
                # all colors taken
                color_appear_loss = prob.view([B, -1, H * W]).max(dim=2)[0].mean()
                # per-pixel, higher confidence, reduce entropy of per-pixel color distribution
                conf_loss = (-prob * torch.log(prob + 1e-16)).mean()
                # entire-image, even distribution among all colors, increase entropy of entire-image color distribution
                info_loss = (-prob.mean(dim=[2, 3]) * torch.log(prob.mean(dim=[2, 3]) + 1e-16)).mean()
                recons_loss = self.MSE_loss(img, transformed_img)
                # print(f'batch{batch_idx}, num_colors_batch{num_colors_batch}, '
                #       f'max_index{torch.max(index_map.view(B, -1), dim=1)[0].float().mean().item() + 1}, '
                #       f'dataset.num_colors{dataloader.dataset.num_colors[0]}')
                loss = self.opts.ce_ratio * ce_loss + \
                       self.opts.recons_ratio * recons_loss * np.log2(num_colors_batch) + \
                       self.opts.colormax_ratio * -color_appear_loss + \
                       self.opts.conf_ratio * conf_loss + self.opts.info_ratio * -info_loss
                if self.opts.kd_ratio or self.opts.perceptual_ratio:
                    # kd loss
                    kd_loss = self.KD_loss(output, output_target.detach())
                    # perceptual loss
                    perceptual_loss = self.MSE_loss(activation[0], activation[1])
                    loss += self.opts.kd_ratio * kd_loss + self.opts.perceptual_ratio * perceptual_loss
                if self.opts.pixsim_ratio:
                    M = torch.zeros_like(prob).scatter(1, index_map, 1)
                    pixsim_loss = self.pixel_loss(prob, M)
                    loss += self.opts.pixsim_ratio * pixsim_loss if num_colors_batch > 2 ** 3 else 0
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
                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, \tLoss: {losses / (batch_idx + 1):.3f}, '
                      f'Prec: {100. * correct / (correct + miss):.1f}%, Time: {t_epoch:.3f}')
                if self.colorcnn:
                    log = f'ce: {ce_loss.item():.3f}, recons: {recons_loss.item():.3f}, color_appear: ' \
                          f'{color_appear_loss.item():.3f}, conf: {conf_loss.item():.3f}, info: {info_loss.item():.3f}'
                    if self.opts.pixsim_ratio:
                        log += f', pixsim: {pixsim_loss.item():.3f}'
                    print(log)

        handle.remove()

        if self.logdir is not None and self.colorcnn:
            img, target = next(iter(dataloader))
            self.sample_image(img.cuda(), num_colors_batch, quantized_img=target[1][0].cuda(),
                              fname=f'{self.logdir}/imgs/{epoch:03d}_train.png')

        return losses / len(dataloader), correct / (correct + miss)

    def test(self, dataloader, num_colors=None, epoch=None, visualize=False, test_mode='test'):
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
            index = [28, 31, 40, 43, 5, 6]
            image_grid = make_grid(data_quantized[index], nrow=8, normalize=False)
            # Arange images along y-axis
            save_image(image_grid, f'{self.logdir}/imgs/batch_{num_colors}.png', normalize=False)

            if self.colorcnn:
                og_img = data[i].cpu().numpy().squeeze().transpose([1, 2, 0])
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
            cam_map = np.sum(activation['classifier'][i] * weight_softmax[pred[i].item()].reshape((-1, 1, 1)), axis=0)
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

        buffer_size_counter, number_of_colors, dataset_size = 0, 0, 0
        self.classifier.eval()
        if self.colorcnn:
            self.colorcnn.eval()
        losses, correct, miss = 0, 0, 0
        t0 = time.time()

        if visualize:
            if hasattr(self.classifier, 'features'):
                classifier_handle = self.classifier.features.register_forward_hook(classifier_activation_hook)
            else:
                classifier_handle = self.classifier.layer4.register_forward_hook(classifier_activation_hook)
            if hasattr(self.classifier, 'classifier'):
                classifier_layer = self.classifier.classifier
            else:
                classifier_layer = self.classifier.fc
            if isinstance(classifier_layer, nn.Sequential):
                classifier_layer = classifier_layer[-1]
            weight_softmax = classifier_layer.weight.detach().cpu().numpy()
            if self.colorcnn:
                colorcnn_handle = self.colorcnn.base.register_forward_hook(auto_encoder_activation_hook)

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(), target.cuda()
            data_og = data.clone()
            B, C, H, W = data.shape
            # quantization first
            if self.sample_trans is None:
                if self.colorcnn:
                    with torch.no_grad():
                        data_quantized, prob, _ = self.colorcnn(data, num_colors, mode=test_mode)
                        data_quantized = data_quantized.clamp(0, 1)
                else:
                    data_quantized = data

                if self.adversarial:
                    data_quantized, _, adv_success = self.adversarial(self.fclassifier, data_quantized, target,
                                                                      epsilons=self.epsilon)
            # adversarial first
            else:
                if self.adversarial:
                    data, _, adv_success = self.adversarial(self.fclassifier, data, target, epsilons=self.epsilon)
                if self.colorcnn:
                    with torch.no_grad():
                        data_quantized, prob, _ = self.colorcnn(data, num_colors, mode=test_mode)
                        data_quantized = data_quantized.clamp(0, 1)
                else:
                    img_list = []
                    for i in range(B):
                        img_list.append(self.sample_trans(data[i].cpu()).to(data.device))
                    data_quantized = torch.stack(img_list, dim=0)

            recons_loss_list.append(self.MSE_loss(data, data_quantized).item())

            with torch.no_grad():
                output = self.classifier(self.norm(data_quantized))
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            # if self.adversarial:
            #     assert adv_success.int().sum().item() == target.shape[0] - pred.eq(target).sum().item()
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
                if batch_idx == 1:
                    i = 6
                    visualize_img(i)
                    print(f'pred: {pred[i]}, target: {target[i]}, '
                          f'conf: {torch.softmax(output, 1)[i][target[i]] * 100:.1f}%')
                    break

        print(f'Test, loss: {losses / (len(dataloader) + 1):.3f}, prec: {100. * correct / (correct + miss):.2f}%, '
              f'time: {time.time() - t0:.1f}, recons loss: {np.mean(recons_loss_list):.3f}')
        if self.colorcnn:
            print(f'Average number of colors per image: {number_of_colors / dataset_size}; \n'
                  f'Average image size: {buffer_size_counter / dataset_size:.1f}; '
                  f'Bit per pixel: {buffer_size_counter / dataset_size / H / W:.3f}')

        if visualize:
            classifier_handle.remove()
            if self.colorcnn:
                colorcnn_handle.remove()

        if epoch is not None and self.logdir is not None:
            it = iter(dataloader)
            img, _ = next(it)
            img, _ = next(it)
            self.sample_image(img.cuda(), num_colors, fname=f'{self.logdir}/imgs/{epoch:03d}_{num_colors}.png')

        return losses / len(dataloader), correct / (correct + miss)

    def sample_image(self, img, num_colors, fname, quantized_img=None, nrow=4):
        if self.colorcnn:
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
