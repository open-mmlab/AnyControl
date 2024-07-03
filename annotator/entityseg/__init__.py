# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import sys
sys.path.insert(1, os.getcwd())

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

from annotator.util import annotator_ckpts_path


model_url = "https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/resolve/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x.pth"


def make_colors():
    from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
    colors = []
    for cate in COCO_CATEGORIES:
        colors.append(cate["color"])
    return colors


class EntitysegDetector:

    def __init__(self, confidence_threshold=0.5):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        workdir = os.getcwd()
        config_file = f"{workdir}/annotator/entityseg/configs/cropformer_hornet_3x.yaml"
        model_path = f'{annotator_ckpts_path}/CropFormer_hornet_3x_03823a.pth'
        # Authentication required
        # if not os.path.exists(model_path):
        #     from basicsr.utils.download_util import load_file_from_url
        #     load_file_from_url(model_url, model_dir=annotator_ckpts_path)

        cfg.merge_from_file(config_file)
        opts = ['MODEL.WEIGHTS', model_path]
        cfg.merge_from_list(opts)
        cfg.freeze()

        self.confidence_threshold = confidence_threshold

        self.colors = make_colors()
        self.demo = VisualizationDemo(cfg)


    def __call__(self, image): 
        predictions = self.demo.run_on_image(image)
        ##### color_mask
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores
        
        # select by confidence threshold
        selected_indexes = (pred_scores >= self.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]
        selected_masks  = pred_masks[selected_indexes]
        _, m_H, m_W = selected_masks.shape
        mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

        # rank
        selected_scores, ranks = torch.sort(selected_scores)
        ranks = ranks + 1
        for index in ranks:
            mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)
        unique_mask_id = np.unique(mask_id)

        color_mask = np.zeros(image.shape, dtype=np.uint8)
        for count in unique_mask_id:
            if count == 0:
                continue
            color_mask[mask_id==count] = self.colors[count % len(self.colors)]
        
        return color_mask
