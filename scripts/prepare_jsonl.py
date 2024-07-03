import os
import sys
import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AnyControl Training Data')
    parser.add_argument('--dataset', type=str, choices=["MultiGen-20M", "COCO", "OpenImages"], default="MultiGen-20M")
    parser.add_argument('--data_root', type=str, default="./datasets")
    args = parser.parse_args()

    conditions = ["canny", "hed", "depth", "seg", "openpose"]

    # MultiGen-20M
    if args.dataset == "MultiGen-20M":
        save_path = os.path.join(args.data_root, "MultiGen-20M/anycontrol_annotations.jsonl")
        cond_path = os.path.join(args.data_root, "MultiGen-20M/json_files/aesthetics_plus_all_group_{cond}_all.json")
        data_samples = {}
        for cond in conditions:
            for line in tqdm(open(cond_path.format(cond=cond)).read().splitlines()):
                ori_dp = json.loads(line)
                source = ori_dp['source']
                if source not in data_samples:
                    data_samples[source] = ori_dp
                else:
                    data_samples[source][f"control_{cond}"] = ori_dp[f"control_{cond}"]
        with open(save_path, "w") as fid:
            for key, value in data_samples.items():
                fid.write(json.dumps(value)+'\n')

    # COCO
    if args.dataset == "COCO":
        caption_path = os.path.join(args.data_root, "MSCOCO/annotations/captions_train2017.json")
        coco = json.load(open(caption_path))
        coco_annos = coco["annotations"]
        prompts = {"%012d" % int(ann["image_id"]): ann["caption"] for ann in coco_annos}

        save_path = os.path.join(args.data_root, "MSCOCO/anycontrol_annotations.jsonl")
        cond_root = os.path.join(args.data_root, "MSCOCO/conditions")
        data_samples = {}
        for cond in ["mask", "inpaint"]:
            for filename in os.listdir(os.path.join(cond_root, cond)):
                img_id = filename.split('.')[0]
                if img_id not in data_samples:
                    source = filename.split('_')[0]+'.jpg'
                    data_samples[img_id] = dict(
                        source=source,
                        prompt=prompts[img_id.split('_')[0]],
                    ) 
                data_samples[img_id][cond] = os.path.join(cond, filename)
        for cond in conditions:
            for filename in os.listdir(os.path.join(cond_root, cond)):
                img_id = filename.split('.')[0]
                if img_id in data_samples:
                    origin_filename = filename.split('_')[0]+'.jpg'
                    data_samples[img_id][f"control_{cond}"] = os.path.join(cond, filename)
                    data_samples[img_id][f"origin_{cond}"] = os.path.join(f"origin_{cond}", origin_filename)
        with open(save_path, "w") as fid:
            for key, value in data_samples.items():
                fid.write(json.dumps(value)+'\n')
    
    # OpenImages
    if args.dataset == "OpenImages":
        save_path = os.path.join(args.data_root, "OpenImages/anycontrol_annotations.jsonl")
        cond_root = os.path.join(args.data_root, "OpenImages/conditions")
        blip_caption_root = os.path.join(args.data_root, "OpenImages/conditions/blipcaption")
        data_samples = {}
        for cond in ["mask", "inpaint"]:
            for filename in os.listdir(os.path.join(cond_root, cond)):
                img_id = filename.split('.')[0]
                if img_id not in data_samples:
                    source = filename.split('_')[0]+'.jpg'
                    data_samples[img_id] = dict(
                        source=source,
                        prompt=open(os.path.join(blip_caption_root, filename.split('_')[0]+'.txt')).read(),
                    ) 
                data_samples[img_id][cond] = os.path.join(cond, filename)
        for cond in conditions:
            for filename in os.listdir(os.path.join(cond_root, cond)):
                img_id = filename.split('.')[0]
                if img_id in data_samples:
                    origin_filename = filename.split('_')[0]+'.jpg'
                    data_samples[img_id][f"control_{cond}"] = os.path.join(cond, filename)
                    data_samples[img_id][f"origin_{cond}"] = os.path.join(f"origin_{cond}", origin_filename)
        with open(save_path, "w") as fid:
            for key, value in data_samples.items():
                fid.write(json.dumps(value)+'\n')
