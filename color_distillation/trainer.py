import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import foolbox as fb
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO
from color_distillation.loss import LSR_loss, KD_loss
from color_distillation.models.alexnet import AlexNet
from color_distillation.models.vgg import VGG
from color_distillation.models.resnet import ResNet
from color_distillation.utils.image_utils import Normalize, DeNormalize


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, classifier, colorcnn=None, label_smooth=0.0, adversarial=None, epsilon=2,
                 mean_var=None, kd_ratio=0.0, perceptual_ratio=0.0,
                 colormax_ratio=0.0, colorvar_ratio=0.0,
                 conf_ratio=0.0, info_ratio=0.0, sample_method=None, sample_trans=None):
        super(BaseTrainer, self).__init__()
        self.classifier = classifier
        if label_smooth:
            self.CE_loss = LSR_loss(label_smooth)
        else:
            self.CE_loss = nn.CrossEntropyLoss()
        self.KD_loss = KD_loss()
        self.MSE_loss = nn.MSELoss()
        self.colorcnn = colorcnn
        self.kd_ratio, self.perceptual_ratio, self.colormax_ratio, self.colorvar_ratio, self.conf_ratio, self.info_ratio = \
            kd_ratio, perceptual_ratio, colormax_ratio, colorvar_ratio, conf_ratio, info_ratio
        self.sample_method = sample_method
        if colorcnn is not None:
            self.sample_method = 'colorcnn'
        self.normalize = Normalize(mean_var[0], mean_var[1])
        self.denormalize = DeNormalize(mean_var[0], mean_var[1])
        # self.denormalize = Normalize(-mean_var[0] / mean_var[1], 1 / mean_var[1])
        if adversarial:
            self.fclassifier = fb.PyTorchModel(self.classifier, bounds=(0, 1),
                                               preprocessing=dict(mean=mean_var[0], std=mean_var[1], axis=-3))
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
        self.sample_trans = sample_trans

    def train(self, epoch, data_loader, optimizer, num_colors=None, log_interval=100, cyclic_scheduler=None, ):
        if self.colorcnn:
            self.colorcnn.train()
            self.classifier.eval()
        else:
            self.classifier.train()
        activation = []

        def cnn_activation_hook(module, input, output):
            activation.append(output)

        if isinstance(self.classifier, AlexNet):
            handle = self.classifier.features[-2].register_forward_hook(cnn_activation_hook)
        elif isinstance(self.classifier, VGG):
            handle = self.classifier.features.register_forward_hook(cnn_activation_hook)
        elif isinstance(self.classifier, ResNet):
            handle = self.classifier.layer4.register_forward_hook(cnn_activation_hook)
        else:
            raise Exception
        losses, correct, miss = 0, 0, 0
        num_colors_batch = num_colors
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            activation = []
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if self.colorcnn:
                # negative #color refers to multiple setting one model, thus needs random setting during training
                if num_colors_batch < 0 or (num_colors < 0 and batch_idx % 20 == 0):
                    num_colors_batch = 2 ** np.random.randint(1, int(np.log2(-num_colors)) + 1)
                transformed_img, prob, color_palette = self.colorcnn(data, num_colors_batch)
                output = self.classifier(transformed_img)
                if self.perceptual_ratio or self.kd_ratio:
                    output_target = self.classifier(data)
            else:
                output = self.classifier(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.CE_loss(output, target)
            if self.colorcnn:
                B, _, H, W = data.shape
                # all colors taken
                color_appear_loss = prob.view([B, -1, H * W]).max(dim=2)[0].mean()
                # color palette choose different colors
                color_var_loss = color_palette.squeeze().std(dim=2).mean()
                # per-pixel, higher confidence, reduce entropy of per-pixel color distribution
                conf_loss = (-prob * torch.log(prob + 1e-16)).mean()
                # entire-image, even distribution among all colors, increase entropy of entire-image color distribution
                info_loss = (-prob.mean(dim=[2, 3]) * torch.log(prob.mean(dim=[2, 3]) + 1e-16)).mean()
                loss += self.colormax_ratio * -color_appear_loss + self.colorvar_ratio * -color_var_loss + \
                        self.conf_ratio * conf_loss + self.info_ratio * -info_loss
                if self.perceptual_ratio or self.kd_ratio:
                    # kd loss
                    kd_loss = self.KD_loss(output, output_target.detach())
                    # perceptual loss
                    perceptual_loss = self.MSE_loss(activation[0], activation[1])
                    loss += self.kd_ratio * kd_loss + self.perceptual_ratio * perceptual_loss

            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))
        if self.colorcnn:
            print(f'color_appear_loss: {color_appear_loss.item():.3f}, conf_loss: {conf_loss.item():.3f}, '
                  f'info_loss: {info_loss.item():.3f}')

        handle.remove()

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, num_colors=None, visualize=False, test_mode='test'):
        activation = {}

        def classifier_activation_hook(module, input, output):
            activation['classifier'] = output.cpu().detach().numpy()

        def auto_encoder_activation_hook(module, input, output):
            activation['auto_encoder'] = output.cpu().detach().numpy()

        def visualize_img(i):
            def save_img_batch(img):
                index = [4, 13, 23, 24, 26, 37, 46, 55]
                image_grid = make_grid(img[index], nrow=1, normalize=True)
                # Arange images along y-axis
                save_image(image_grid, 'batch_imgs.png', normalize=False)

            if self.colorcnn:
                save_img_batch(data_colorcnn)
                og_img = data_og[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(og_img)
                plt.show()
                og_img = Image.fromarray((og_img * 255).astype('uint8')).resize((512, 512))
                og_img.save('og_img.png')

                downsampled_img = data_colorcnn[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(downsampled_img)
                plt.show()
                downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8')).resize((512, 512))
                downsampled_img.save('colorcnn.png')

                fig = plt.figure(frameon=False)
                fig.set_size_inches(2, 2)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(np.linalg.norm(activation['auto_encoder'][i], axis=0), cmap='viridis')
                plt.savefig('auto_encoder.png')
                plt.show()
                # index map
                plt.imshow(M[i, 0].cpu().numpy(), cmap='Blues')
                plt.savefig("M.png", bbox_inches='tight')
                plt.show()
            else:
                save_img_batch(data_og)
                downsampled_img = data_og[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(downsampled_img)
                plt.show()
                downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8')).resize((512, 512))
                downsampled_img.save(self.sample_method + '.png')
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
            cv2.imwrite(self.sample_method + '_cam.png', cam_result)
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
            classifier_layer = self.classifier.classifier
            if isinstance(classifier_layer, nn.Sequential):
                classifier_layer = classifier_layer[-1]
            weight_softmax = classifier_layer.weight.detach().cpu().numpy()
            if self.colorcnn:
                colorcnn_handle = self.colorcnn.base_global.register_forward_hook(auto_encoder_activation_hook)

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            data_og = data.clone()
            B, C, H, W = data.shape
            # quantization first
            if self.sample_trans is None:
                if self.colorcnn:
                    with torch.no_grad():
                        data = self.normalize(data)
                        data, prob, _ = self.colorcnn(data, num_colors, mode=test_mode)
                        data = self.denormalize(data)
                        data_colorcnn = data.clone()
                # attack
                if self.adversarial:
                    data, _, _ = self.adversarial(self.fclassifier, data, target, epsilons=self.epsilon)
            # adversarial first
            else:
                # attack
                if self.adversarial:
                    data, _, _ = self.adversarial(self.fclassifier, data, target, epsilons=self.epsilon)
                if self.colorcnn:
                    with torch.no_grad():
                        data = self.normalize(data)
                        data, prob, _ = self.colorcnn(data, num_colors, mode=test_mode)
                        data = self.denormalize(data)
                        data_colorcnn = data.clone()
                else:
                    img_list = []
                    for i in range(B):
                        img_list.append(self.sample_trans(data[i].cpu()).to(data.device))
                    data = torch.stack(img_list, dim=0)

            data = self.normalize(data)
            with torch.no_grad():
                output = self.classifier(data)
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
                    downsampled_img = data_colorcnn[i].cpu().numpy().squeeze().transpose([1, 2, 0])
                    downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8'))

                    png_buffer = BytesIO()
                    downsampled_img.save(png_buffer, "PNG")
                    buffer_size_counter += png_buffer.getbuffer().nbytes
                    dataset_size += 1
            # plotting
            if visualize:
                if batch_idx == 0:  # 0,1,4,
                    visualize_img(0)
                    break

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))
        if self.colorcnn:
            print(f'Average number of colors per image: {number_of_colors / dataset_size}; \n'
                  f'Average image size: {buffer_size_counter / dataset_size:.1f}; '
                  f'Bit per pixel: {buffer_size_counter / dataset_size / H / W:.3f}')

        if visualize:
            classifier_handle.remove()
            if self.colorcnn:
                colorcnn_handle.remove()

        return losses / len(test_loader), correct / (correct + miss)
