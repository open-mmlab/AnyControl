import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "annotator/entityseg"))
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.hed import HEDdetector
from annotator.midas import MidasDetector
from annotator.entityseg import EntitysegDetector
from annotator.openpose import OpenposeDetector


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AnyControl Training Data')
    parser.add_argument('--dataset', type=str, choices=["COCO", "OpenImages"], default="COCO")
    parser.add_argument('--data_root', type=str, default="./datasets")
    args = parser.parse_args()

    condition_types = ["canny", "hed", "depth", "seg", "openpose"]
    
    apply_canny = CannyDetector()
    apply_hed = HEDdetector()
    apply_midas = MidasDetector()
    apply_seg = EntitysegDetector()
    apply_openpose = OpenposeDetector()
    
    processors = {
        "canny": apply_canny,
        "hed": apply_hed,
        "depth": apply_midas,
        "seg": apply_seg,
        "openpose": apply_openpose,
    }

    if args.dataset == "COCO":
        root = os.path.join(args.data_root, "MSCOCO/train2017")
        save_root = os.path.join(args.data_root, "MSCOCO/conditions/origin_{cond}")
        for cond in condition_types:
             os.makedirs(save_root.format(cond=cond), exist_ok=True)

        for filename in tqdm(sorted(os.listdir(root))):
             img = HWC3(cv2.imread(os.path.join(root, filename)))
             for cond in condition_types:
                 cond_ret = processors[cond](img)
                 # sometimes there is no person in the image
                 if cond_ret.sum() == 0:
                     continue
                 cv2.imwrite(os.path.join(save_root.format(cond=cond), filename), cond_ret)

        root = os.path.join(args.data_root, "MSCOCO/conditions/inpaint")
        save_root = os.path.join(args.data_root, "MSCOCO/conditions/{cond}")
        for cond in condition_types:
             os.makedirs(save_root.format(cond=cond), exist_ok=True)

        for filename in tqdm(sorted(os.listdir(root))):
             img = HWC3(cv2.imread(os.path.join(root, filename)))
             for cond in condition_types:
                 cond_ret = processors[cond](img)
                 # sometimes there is no person in the image
                 if cond_ret.sum() == 0:
                     continue
                 cv2.imwrite(os.path.join(save_root.format(cond=cond), filename), cond_ret)
                 
    if args.dataset == "OpenImages":
        root = os.path.join(args.data_root, "OpenImages/train")
        save_root = os.path.join(args.data_root, "OpenImages/conditions/origin_{cond}")
        for cond in condition_types:
             os.makedirs(save_root.format(cond=cond), exist_ok=True)

        for filename in tqdm(sorted(os.listdir(root))):
             img = HWC3(cv2.imread(os.path.join(root, filename)))
             for cond in condition_types:
                 cond_ret = processors[cond](img)
                 # sometimes there is no person in the image
                 if cond_ret.sum() == 0:
                     continue
                 cv2.imwrite(os.path.join(save_root.format(cond=cond), filename), cond_ret)

        root = os.path.join(args.data_root, "OpenImages/conditions/inpaint")
        save_root = os.path.join(args.data_root, "OpenImages/conditions/{cond}")
        for cond in condition_types:
             os.makedirs(save_root.format(cond=cond), exist_ok=True)

        for filename in tqdm(sorted(os.listdir(root))):
             img = HWC3(cv2.imread(os.path.join(root, filename)))
             for cond in condition_types:
                 cond_ret = processors[cond](img)
                 # sometimes there is no person in the image
                 if cond_ret.sum() == 0:
                     continue
                 cv2.imwrite(os.path.join(save_root.format(cond=cond), filename), cond_ret)
