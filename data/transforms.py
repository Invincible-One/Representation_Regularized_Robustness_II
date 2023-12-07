from PIL import Image

import numpy as np

from torchvision.transforms import functional as F



class GrayToRGB:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                return np.stack([img, img, img], axis=-1)
            return img
        else:
            if img.mode == 'L':
                img = img.convert('RGB')
            return img


class NumpyToPIL:
    def __call__(self, arr):
        return Image.fromarray((arr * 255).astype(np.uint8))
