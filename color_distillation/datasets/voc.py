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

CLASSES = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
           'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
           'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}


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


class _VOCBase(VisionDataset):
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
            self,
            root: str,
            year: str = "2012_aug",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            color_quantize: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, transforms=transforms)

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
        if year == '2012_aug' and image_set == 'train' and self._SPLITS_DIR == 'Segmentation':
            file_names, excluded_file_names = [], []
            with open(os.path.join(self.voc_root, "ImageSets/Segmentation/val.txt"), "r") as f:
                excluded_file_names = [x.strip() for x in f.readlines()]
            for file in os.listdir(f"{self.voc_root}/SegmentationClassAug"):
                file_names.append(file.strip('.png'))
            file_names = set(file_names) - set(excluded_file_names)
            file_names = sorted(list(file_names))
        else:
            with open(f"{self.voc_root}/ImageSets/{self._SPLITS_DIR}/{image_set}.txt", "r") as f:
                file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.voc_root, "JPEGImages", x + ".jpg") for x in file_names]
        self.targets = [f"{self.voc_root}/{self._TARGET_DIR}" \
                        f"{'Aug' if year == '2012_aug' and self._SPLITS_DIR == 'Segmentation' else ''}/{x}" \
                        f"{self._TARGET_FILE_EXT}" for x in file_names]
        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)


class VOCSegmentation(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    @classmethod
    def decode_target(self, mask):
        """decode semantic mask to RGB image"""
        return self.cmap[mask]

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
            index_map = torch.tensor(index_map.reshape(H, W)).long().unsqueeze(0)
            quantized_img = self.to_tensor(quantized_img)

        if isinstance(image, Image.Image):
            image = self.to_tensor(image)
            target = torch.from_numpy(np.array(target)).long()

        if self.color_quantize is not None:
            return image, (target, (quantized_img, index_map))
        else:
            return image, target


class VOCClassification(_VOCBase):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
        image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
            ``year=="2007"``, can also be ``"test"``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        image = Image.open(self.images[index]).convert("RGB")
        anno = self.parse_voc_xml(ET_parse(self.targets[index]).getroot())
        target = [CLASSES[obj['name']] for obj in anno['annotation']['object']]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if self.color_quantize is not None:
            self.color_quantize.num_colors = self.num_colors[0]
            quantized_img = self.color_quantize(image)
            H, W, C = np.array(quantized_img).shape
            palette, index_map = np.unique(np.array(quantized_img).reshape([H * W, C]), axis=0, return_inverse=True)
            index_map = torch.tensor(index_map.reshape(H, W)).long().unsqueeze(0)
            quantized_img = self.to_tensor(quantized_img)
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
    import color_distillation.utils.ext_transforms as exT


    def visualize(image, target=None):
        plt.imshow(image if isinstance(image, Image.Image) else image.numpy().transpose([1, 2, 0]))
        plt.show()
        if target is not None:
            plt.imshow(target)
            plt.show()


    dataset = VOCClassification('/home/houyz/Data/pascal_VOC', color_quantize=T.MedianCut(),
                                transform=T.Compose([T.Resize(112, max_size=128), T.RandomCrop(112, pad_if_needed=True),
                                                     T.MedianCut(32), T.PNGCompression(), ]))
    img, tgt = dataset.__getitem__(0)
    visualize(img)

    dataset = VOCSegmentation('/home/houyz/Data/pascal_VOC', color_quantize=T.MedianCut(),
                              transforms=exT.ExtCompose([T.MedianCut(32), T.PNGCompression(),
                                                         exT.ExtRandomScale([0.5, 2.0]), ]))
    img, tgt = dataset.__getitem__(0)
    visualize(img, tgt[0])
    pass
