import cv2
import numpy as np
import torch

from einops import rearrange
from .api import MiDaSInference


class MidasDetector:
    def __init__(self):
        self.model = MiDaSInference(model_type="dpt_hybrid").cuda()

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3

        oh, ow = input_image.shape[:2]
        nh = oh // 32 * 32
        nw = ow // 32 * 32
        input_image = cv2.resize(input_image, (nw, nh))
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_image = cv2.resize(depth_image, (nw, nh))

            return depth_image
