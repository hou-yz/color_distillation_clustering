import os
import numpy as np
import torch
import collections
from torchvision.datasets.vision import VisionDataset
from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import torch.multiprocessing as mp
import color_distillation.utils.transforms as T

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2012_aug': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class VOC(VisionDataset):
    def __init__(
            self,
            root: str,
            year: str = "2012_aug",
            image_set: str = "train",
            download: bool = False,
            transforms: Optional[Callable] = None,
            color_quantize: Optional[Callable] = None,
    ):
        super().__init__(root, transforms=transforms)

        # shared memory support
        # https://discuss.pytorch.org/t/dataloader-resets-dataset-state/27960/4
        # https://discuss.pytorch.org/t/how-to-share-data-among-dataloader-processes-to-save-memory/108772
        shared_array_base = mp.Array('i', 1)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array[0] = 16
        self.num_colors = shared_array
        self.color_quantize = color_quantize
        self.to_tensor = T.ToTensor()

        # download
        image_set = verify_str_arg(image_set, "image_set", ["train", "trainval", "val"])
        dataset_year_dict = DATASET_YEAR_DICT[year]
        self.voc_root = os.path.join(self.root, dataset_year_dict["base_dir"])
        if download:
            download_and_extract_archive(dataset_year_dict["url"], self.root,
                                         filename=dataset_year_dict["filename"],
                                         md5=dataset_year_dict["md5"])
        if not os.path.isdir(self.voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # use the detection/classification partition
        # find file names
        if year == '2012_aug' and image_set == 'train':
            file_names, excluded_file_names = [], []
            with open(os.path.join(self.voc_root, "ImageSets/Segmentation/val.txt"), "r") as f:
                excluded_file_names = [x.strip() for x in f.readlines()]
            for file in os.listdir(f"{self.voc_root}/SegmentationClassAug"):
                file_names.append(file.strip('.png'))
            file_names = set(file_names) - set(excluded_file_names)
            file_names = sorted(list(file_names))
        else:
            with open(f"{self.voc_root}/ImageSets/Segmentation/{image_set}.txt", "r") as f:
                file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.voc_root, "JPEGImages", x + ".jpg") for x in file_names]
        self.targets = [f"{self.voc_root}/SegmentationClass{'Aug' if year == '2012_aug' else ''}/{x}.png"
                        for x in file_names]
        assert len(self.images) == len(self.targets)

    @classmethod
    def decode_target(self, mask):
        """decode semantic mask to RGB image"""
        return self.cmap[mask]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.color_quantize is not None:
            self.color_quantize.num_colors = self.num_colors[0]
            quantized_img = self.color_quantize(image)
            H, W, C = np.array(quantized_img).shape
            palette, index_map = np.unique(np.array(quantized_img).reshape([H * W, C]), axis=0, return_inverse=True)
            index_map = Image.fromarray(index_map.reshape(H, W).astype(np.uint8))
            quantized_img, index_map = self.to_tensor(quantized_img), (self.to_tensor(index_map) * 255).round().long()

        if isinstance(image, Image.Image):
            image = self.to_tensor(image)
            target = torch.from_numpy(np.array(target)).long()

        if self.color_quantize is not None:
            return image, (target, (quantized_img, index_map))
        else:
            return image, target


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import color_distillation.utils.ext_transforms as exT


    def visualize(image, target):
        plt.imshow(target)
        plt.show()
        plt.imshow(image if isinstance(image, Image.Image) else image.numpy().transpose([1, 2, 0]))
        plt.show()


    dataset = VOC('/home/houyz/Data/pascal_VOC', color_quantize=T.MedianCut(),
                  transforms=exT.ExtCompose([T.MedianCut(2), T.PNGCompression(), exT.ExtRandomScale([0.5, 2.0]), ]))
    img, tgt = dataset.__getitem__(0)
    visualize(img, tgt[0])
    pass
