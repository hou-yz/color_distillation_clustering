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
}

CLASSES = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
           'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
           'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}


class VOC(VisionDataset):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            color_quantize: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

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
        with open(f"{self.voc_root}/ImageSets/Main/{image_set}.txt", "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.voc_root, "JPEGImages", x + ".jpg") for x in file_names]
        self.targets = [f"{self.voc_root}/Annotations/{x}.xml"
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
        anno = self.parse_voc_xml(ET_parse(self.targets[index]).getroot())
        target = [CLASSES[obj['name']] for obj in anno['annotation']['object']]

        if self.transform is not None:
            image = self.transform(image)

        if self.color_quantize is not None:
            self.color_quantize.num_colors = self.num_colors[0]
            quantized_img = self.color_quantize(image)
            H, W, C = np.array(quantized_img).shape
            palette, index_map = np.unique(np.array(quantized_img).reshape([H * W, C]), axis=0, return_inverse=True)
            index_map = Image.fromarray(index_map.reshape(H, W).astype(np.uint8))
            quantized_img, index_map = self.to_tensor(quantized_img), (self.to_tensor(index_map) * 255).round().long()
        if isinstance(image, Image.Image):
            image = self.to_tensor(image)

        target = torch.zeros([20]).scatter(0, torch.tensor(target).unique() - 1, 1)

        if self.color_quantize is not None:
            return image, (target, (quantized_img, index_map))
        else:
            return image, target

    def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def visualize(image):
        plt.imshow(image if isinstance(image, Image.Image) else image.numpy().transpose([1, 2, 0]))
        plt.show()


    dataset = VOC('/home/houyz/Data/pascal_VOC', color_quantize=T.MedianCut(),
                  transform=T.Compose([T.RandomResizedCrop(112), T.RandomHorizontalFlip(), ]))
    img, target = dataset.__getitem__(0)
    visualize(img)
    pass
