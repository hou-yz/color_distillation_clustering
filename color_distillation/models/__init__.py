from .alexnet import AlexNet
from .vgg import VGG16
from .resnet import ResNet18, ResNet50, ResNet152
from .deeplabv3 import DeepLabV3

custom_factory = {
    'alexnet': AlexNet,
    'vgg16': VGG16,
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet152': ResNet152,
    'deeplab': DeepLabV3
}

from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16_bn
from torchvision.models.resnet import resnet18, resnet50, resnet152

torchvision_factory = {
    'alexnet': alexnet,
    'vgg16': vgg16_bn,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet152': resnet152,
}


def names():
    return sorted(custom_factory.keys())


def create(name, out_channel, pretrained=False):
    """
    Create a model instance.
    """
    if out_channel in [10, 100, 200, 21]:
        # use custom models
        if name not in custom_factory:
            raise KeyError("Unknown model:", name)
        return custom_factory[name](out_channel)
    else:
        if name not in torchvision_factory:
            raise KeyError("Unknown model:", name)
        model = torchvision_factory[name](pretrained=pretrained)
        if out_channel != 1000:
            import torch.nn as nn
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, out_channel)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Linear):
                    model.classifier = nn.Linear(model.classifier.in_features, out_channel)
                elif isinstance(model.classifier, nn.Sequential):
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, out_channel)
        return model
