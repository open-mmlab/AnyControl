import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'rayleigh'))

import numpy as np
from skimage.color import rgb2lab
from .rayleigh import Palette
from .rayleigh.util import histogram_colors_strict, smooth_histogram, color_hist_to_palette_image


class CIELabDetector:

    MAX_DIMENSION = 240 + 1

    def __init__(self, sigma=10, num_hues=11, num_light=5, num_sat=5):
        self.sigma = sigma
        self.palette = Palette(num_hues=num_hues, light_range=num_light, sat_range=num_sat)

    def __call__(self, img):
        # Handle grayscale and RGBA images.
        # TODO: Should be smarter here in the future, but for now simply remove
        # the alpha channel if present.
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.ndim == 4:
            img = img[:, :, :3]
        img = img[:,:,:3]

        h, w, d = tuple(img.shape)
        h_stride = int(h / self.MAX_DIMENSION + 1)
        w_stride = int(w / self.MAX_DIMENSION + 1)
        img = img[::h_stride, ::w_stride, :]

        # Convert to L*a*b colors.
        h, w, d = img.shape
        lab_array = rgb2lab(img).reshape((h * w, d))

        # compute hist
        hist = histogram_colors_strict(lab_array, self.palette)
        hist = smooth_histogram(hist, self.palette, self.sigma)
        return hist

    def hist_to_palette(self, hist):
        # hist to image
        plt = color_hist_to_palette_image(hist, self.palette)
        return (plt * 255).astype(np.uint8)
