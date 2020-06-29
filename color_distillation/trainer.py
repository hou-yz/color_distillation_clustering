import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO
from color_distillation.loss.entropy_loss import HLoss


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, classifier=None, denormalizer=None,
                 color_appear_ratio=None, confidence_ratio=None, gamma=None, sample_method=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.CE_loss = criterion
        self.IE_loss = HLoss()
        self.classifier = classifier
        self.denormalizer = denormalizer
        self.color_appear_ratio = color_appear_ratio
        self.confidence_ratio = confidence_ratio
        self.gamma = gamma
        self.reconsturction_loss = nn.MSELoss()
        self.sample_method = sample_method
        if classifier is not None:
            self.color_cnn = True
            self.sample_method = 'colorcnn'
        else:
            self.color_cnn = False

    def train(self, epoch, data_loader, optimizer, num_colors=None, log_interval=100, cyclic_scheduler=None, ):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if self.color_cnn:
                # negative #color refers to multiple setting one model, thus needs random setting during training
                if num_colors < 0:
                    num_colors = 2 ** np.random.randint(1, int(np.log2(-num_colors)))
                transformed_img, prob, color_palette = self.model(data, num_colors)
                # regularization
                B, _, H, W = data.shape
                prob_max, _ = torch.max(prob.view([B, num_colors, -1]), dim=2)
                avg_max = torch.mean(prob_max)
                output = self.classifier(transformed_img)
            else:
                output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.CE_loss(output, target)
            if self.color_cnn:
                color_appear_loss = np.log2(num_colors) * (1 - avg_max)
                confidence_loss = self.IE_loss(prob)
                loss += self.color_appear_ratio * color_appear_loss + \
                        self.confidence_ratio * confidence_loss

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
        if self.color_cnn:
            print(f'color_appear_loss: {color_appear_loss.item():.3f}, confidence_loss: {confidence_loss.item():.3f}')

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, num_colors=None, visualize=False):
        activation = {}

        def classifier_activation_hook(module, input, output):
            activation['classifier'] = output.cpu().detach().numpy()

        def auto_encoder_activation_hook(module, input, output):
            activation['auto_encoder'] = output.cpu().detach().numpy()

        def visualize_img(i):
            if self.color_cnn:
                og_img = self.denormalizer(data[i]).cpu().numpy().squeeze().transpose([1, 2, 0])
                plt.imshow(og_img)
                plt.show()
                og_img = Image.fromarray((og_img * 255).astype('uint8')).resize((512, 512))
                og_img.save('og_img.png')

                downsampled_img = self.denormalizer(transformed_img[i]).cpu().numpy().squeeze().transpose(
                    [1, 2, 0])
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
                # plt.imshow(M[i, 0].cpu().numpy(), cmap='Blues')
                # plt.savefig("M.png", bbox_inches='tight')
                # plt.show()
            else:
                downsampled_img = self.denormalizer(data[i]).cpu().numpy().squeeze().transpose([1, 2, 0])
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
            cam_result = cv2.putText(cam_result, '{:.1f}%, {}'.format(
                100 * F.softmax(output, dim=1)[i, target[i]].item(),
                'Success' if pred.eq(target)[i].item() else 'Failure'),
                                     (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                     (0, 255, 0) if pred.eq(target)[i].item() else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(self.sample_method + '_cam.jpg', cam_result)
            plt.imshow(cv2.cvtColor(cam_result, cv2.COLOR_BGR2RGB))
            plt.show()
            # ax.imshow(cam_map, cmap='viridis', aspect='auto')
            # fig.savefig("activation.png", )
            # fig.show()

        buffer_size_counter = 0
        number_of_colors = 0
        dataset_size = 0
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()

        if visualize:
            if self.color_cnn:
                self.classifier.features.register_forward_hook(classifier_activation_hook)
                self.model.base.register_forward_hook(auto_encoder_activation_hook)

                classifier_layer = self.classifier.classifier
                if isinstance(classifier_layer, nn.Sequential):
                    classifier_layer = classifier_layer[-1]
                weight_softmax = classifier_layer.weight.detach().cpu().numpy()
            else:
                self.model.features.register_forward_hook(classifier_activation_hook)

                classifier_layer = self.model.classifier
                if isinstance(classifier_layer, nn.Sequential):
                    classifier_layer = classifier_layer[-1]
                weight_softmax = classifier_layer.weight.detach().cpu().numpy()

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                if self.color_cnn:
                    B, C, H, W = data.shape
                    transformed_img, prob, _ = self.model(data, num_colors, training=False)
                    output = self.classifier(transformed_img)
                    # image file size
                    M = torch.argmax(prob, dim=1, keepdim=True)  # argmax color index map
                    for i in range(target.shape[0]):
                        number_of_colors += len(M[0].unique())
                        downsampled_img = self.denormalizer(transformed_img[i]).cpu().numpy().squeeze().transpose(
                            [1, 2, 0])
                        downsampled_img = Image.fromarray((downsampled_img * 255).astype('uint8'))

                        png_buffer = BytesIO()
                        downsampled_img.save(png_buffer, "PNG")
                        buffer_size_counter += png_buffer.getbuffer().nbytes
                        dataset_size += 1
                else:
                    output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.CE_loss(output, target)
            losses += loss.item()
            # plotting
            if visualize:
                if batch_idx == 2:
                    visualize_img(28)
                    break

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))
        if self.color_cnn:
            print(f'Average number of colors per image: {number_of_colors / dataset_size}; \n'
                  f'Average image size: {buffer_size_counter / dataset_size:.1f}; '
                  f'Bit per pixel: {buffer_size_counter / dataset_size / H / W:.3f}')

        return losses / len(test_loader), correct / (correct + miss)
