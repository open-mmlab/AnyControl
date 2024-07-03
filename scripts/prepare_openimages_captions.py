import os
import sys
import json
import glob
import torch
import argparse

from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AnyControl Training Data')
    parser.add_argument('--data_root', type=str, default="./datasets")
    args = parser.parse_args()

    device = 'cuda'
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

    image_root = os.path.join(args.data_root, "OpenImages/train")
    filenames = sorted(os.listdir(root))

    save_root = os.path.join(args.data_root, "OpenImages/conditions/blipcaption")
    os.makedirs(save_root, exist_ok=True)
    
    for filename in tqdm(filenames):

        raw_image = Image.open(os.path.join(root, filename)).convert("RGB")
        image = raw_image.resize((512, 512))
        
        image_pt = vis_processors["eval"](image).unsqueeze(0).to(device)
    
        results = model.generate({"image": image_pt, "prompt": "Question: Describe this image in detail. Answer: "}, max_length=77)[0]
    
        img_id = filename.split('/')[-1].split('.')[0]
        with open(os.path.join(save_root, img_id+'.txt'), 'w') as fid:
            fid.write(results+'\n')
