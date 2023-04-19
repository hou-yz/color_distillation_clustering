import numpy as np
import cv2
import torchvision.transforms.functional as F
from PIL import Image
from io import BytesIO
from color_distillation.utils.dither.palette import Palette
from color_distillation.utils.dither.dithering import error_diffusion_dithering
from torchvision.transforms import *


class CannyEdge(object):
    def __init__(self):
        pass

    def __call__(self, img, tgt=None):
        # Convert RGB to BGR
        open_cv_img = np.array(img)[:, :, ::-1].copy()
        sampled_img = cv2.Canny(open_cv_img, 100, 200)
        sampled_img = Image.fromarray(sampled_img).convert('RGB')
        if tgt is not None:
            return sampled_img, tgt
        else:
            return sampled_img


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class MedianCut(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img, tgt=None):
        if self.num_colors is not None:
            if not self.dither:
                sampled_img = img.quantize(colors=self.num_colors, method=0).convert('RGB')
            else:
                palette = Palette(img.quantize(colors=self.num_colors, method=0))
                sampled_img = error_diffusion_dithering(img, palette).convert('RGB')
        else:
            sampled_img = img
        if tgt is not None:
            return sampled_img, tgt
        else:
            return sampled_img


class OCTree(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img, tgt=None):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, method=2).convert('RGB')
        else:
            sampled_img = img
        if tgt is not None:
            return sampled_img, tgt
        else:
            return sampled_img


class KMeans(object):
    def __init__(self, num_colors=None, dither=False):
        self.num_colors = num_colors
        self.dither = dither

    def __call__(self, img, tgt=None):
        if self.num_colors is not None:
            sampled_img = img.quantize(colors=self.num_colors, kmeans=2).convert('RGB')
        else:
            sampled_img = img
        if tgt is not None:
            return sampled_img, tgt
        else:
            return sampled_img


class PNGCompression(object):
    def __init__(self, buffer_size_counter=None):
        self.buffer_size_counter = buffer_size_counter

    def __call__(self, img, tgt=None):
        png_buffer = BytesIO()
        img.save(png_buffer, "PNG")
        if self.buffer_size_counter is not None:
            self.buffer_size_counter.update(png_buffer.getbuffer().nbytes)
        if tgt is not None:
            return img, tgt
        else:
            return img


class JpegCompression(object):
    def __init__(self, buffer_size_counter, quality=50):
        self.buffer_size_counter = buffer_size_counter
        self.quality = quality

    def __call__(self, img, tgt=None):
        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, "JPEG", quality=self.quality)
        self.buffer_size_counter.update(jpeg_buffer.getbuffer().nbytes)
        jpeg_buffer.seek(0)
        sampled_img = Image.open(jpeg_buffer)
        if tgt is not None:
            return sampled_img, tgt
        else:
            return sampled_img
