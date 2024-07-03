import os
import random
import cv2
import json
import numpy as np

from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from .util import *


def get_task_groups(exist_tasks):
    task_groups = dict() 
    for task in exist_tasks:
        if task in ['canny', 'hed', 'sketch', 'hedsketch']:
            if 'edge' not in task_groups:
                task_groups['edge'] = []
            task_groups['edge'].append(task)
        if task in ['depth']:
            if 'depth' not in task_groups:
                task_groups['depth'] = []
            task_groups['depth'].append(task)
        if task in ['seg']:
            if 'seg' not in task_groups:
                task_groups['seg'] = []
            task_groups['seg'].append(task)
        if task in ['openpose', 'dwpose']:
            if 'pose' not in task_groups:
                task_groups['pose'] = []
            task_groups['pose'].append(task)
    return task_groups


def diff(a, b):
    return list(set(a) - set(b))


def sample_tasks_with_occlusion(default_tasks, sample):
    exist_tasks = [key.split('_')[-1] for key in sample.keys() if key.startswith('control_')]
    exist_tasks = [task for task in exist_tasks if task in default_tasks]
    task_groups = get_task_groups(exist_tasks)

    group = random.choice(diff(task_groups.keys(), ['pose']))
    task1 = random.choice(task_groups[group]) 
    if 'pose' in task_groups and random.random()<0.75:
        task2 = random.choice(task_groups['pose'])
    else:
        group = random.choice(list(task_groups.keys()))
        task2 = random.choice(task_groups[group]) 
    return [task1, task2]


def sample_tasks_without_occlusion(default_tasks, sample):
    exist_tasks = [key.split('_')[-1] for key in sample.keys() if key.startswith('control_')]
    exist_tasks = [task for task in exist_tasks if task in default_tasks]
    task_groups = get_task_groups(exist_tasks)

    if 'pose' in task_groups and random.random()<0.75:
        group = 'pose'
        task1 = random.choice(task_groups[group])
    else:
        group = random.choice(list(task_groups.keys()))
        task1 = random.choice(task_groups[group]) 

    cand_groups = diff(task_groups.keys(), [group]) 
    if len(cand_groups) == 0:
        task2 = random.choice(task_groups[group]) 
    else:
        group = random.choice(cand_groups)
        task2 = random.choice(task_groups[group]) 
    return [task1, task2]


def get_pose_mask(pose):
    mask = (pose>0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    return 1 - mask


def get_dilated_mask(mask):
    mask = (mask>0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    return mask


def resize_and_crop(image, image_shape=None, shape=None, resolution=512):
    h, w = image.shape[:2]
    nw, nh = w, h
    if image_shape is None:
        if min(h,w) != resolution:
            nw = int(round(resolution / min(h,w) * w))
            nh = int(round(resolution / min(h,w) * h))
            image = cv2.resize(image, (nw, nh))
    else:
       if image.shape[:2] != image_shape[::-1]:
           image = cv2.resize(image, image_shape)
    h, w = image.shape[:2]
    if shape is None:
        x1 = random.randint(0, w-resolution)
        x2 = x1 + resolution
        y1 = random.randint(0, h-resolution)
        y2 = y1 + resolution
    else:
        x1, y1, x2, y2 = shape
    image = image[y1:y2, x1:x2]
    assert image.shape[:2] == (resolution, resolution)
    return image, (x1, y1, x2, y2), (nw, nh)


class CustomDataset(Dataset):

    TEMPLATE = "an image of {}"

    TASK_PROMPT = dict(
        hed='hed',
        canny='canny',
        hedsketch='sketch',
        sketch='sketch',
        depth='depth',
        openpose='pose',
        seg='segmentation',
        clipembedding='semantics',
        color='color',
    )

    def __init__(self,
                 local_tasks,
                 json_files,
                 data_root,
                 image_dir,
                 condition_root,
                 resolution,
                 drop_txt_prob,
                 drop_all_prob,
                 keep_all_local_prob,
                 drop_all_local_prob,
                 drop_each_cond_prob):

        self.local_tasks = local_tasks

        # load from multiple jsons
        datapoints = []
        for json_file, ds in json_files:
            if not os.path.exists(json_file):
                continue
            lines = open(json_file).read().splitlines()
            cnt = 0
            for line in lines:
                if line == '': continue
                dp = json.loads(line)
                dp['dataset'] = ds
                datapoints.append(dp)
                cnt += 1
            print(f"Loaded from {json_file}, {cnt} samples")
        self.data = datapoints 

        self.data_root = data_root
        self.image_dir = image_dir
        self.resolution = resolution
        self.condition_root = condition_root
        self.drop_txt_prob = drop_txt_prob
        self.drop_all_prob = drop_all_prob
        self.keep_all_local_prob = keep_all_local_prob
        self.drop_all_local_prob = drop_all_local_prob
        self.drop_each_cond_prob = drop_each_cond_prob

    def __getitem__(self, index):
        
        # loading: some images are broken
        while True:

            index = index % len(self.data)
            image_id = self.data[index]['source']
            text = self.data[index]['prompt']
            ds = self.data[index]['dataset']
            image_path = os.path.join(self.image_dir[ds], image_id)

            image = cv2.imread(image_path)
            if image is None or text is None:
                index = (index + 1) % len(self.data)
                continue

            image, crop_shape, image_shape = resize_and_crop(image, image_shape=None, shape=None)
 
            # load local controls
            local_conds = []
            local_tasks = []
            local_load_status = []
            if ds in ['coco', 'openimages']:
                tasks = sample_tasks_with_occlusion(self.local_tasks, self.data[index])
                task = tasks[0]
                if 'control_'+task in self.data[index] and 'origin_'+task in self.data[index]:
                    cond_path = os.path.join(self.data_root[ds], self.condition_root[ds], self.data[index]['control_'+task])
                    local_cond = cv2.imread(cond_path)
                    local_cond, _, _ = resize_and_crop(local_cond, image_shape, crop_shape)
                    if task == 'sketch' or task == 'hedsketch':
                        local_cond = 255 - local_cond
                    local_conds.append(local_cond)
                    local_tasks.append(task)
                    local_load_status.append(True)
                else:
                    local_cond = np.zeros_like(image)
                    local_conds.append(local_cond)
                    local_tasks.append(task)
                    local_load_status.append(False)

                task = tasks[1]
                if 'control_'+task in self.data[index] and 'origin_'+task in self.data[index]:
                    cond_path = os.path.join(self.data_root[ds], self.condition_root[ds], self.data[index]['origin_'+task])
                    mask_path = os.path.join(self.data_root[ds], self.condition_root[ds], self.data[index]['mask'])
                    local_cond = cv2.imread(cond_path)
                    local_cond, _, _ = resize_and_crop(local_cond, image_shape, crop_shape)
                    if task == 'sketch' or task == 'hedsketch':
                        local_cond = 255 - local_cond
                    mask = cv2.imread(mask_path) / 255.
                    mask = get_dilated_mask(mask)
                    mask, _, _ = resize_and_crop(mask, image_shape, crop_shape)
                    local_cond = (local_cond * mask).astype(np.uint8)
                    local_conds.append(local_cond)
                    local_tasks.append(task)
                    local_load_status.append(True)
                else:
                    local_cond = np.zeros_like(image)
                    local_conds.append(local_cond)
                    local_tasks.append(task)
                    local_load_status.append(False)
            else:
                tasks = sample_tasks_without_occlusion(self.local_tasks, self.data[index])
                for task in tasks:
                    local_cond = None
                    if 'control_'+task in self.data[index]:
                        cond_path = os.path.join(self.data_root[ds], self.condition_root[ds], self.data[index]['control_'+task])
                        local_cond = cv2.imread(cond_path)
                    if local_cond is not None:
                        local_cond, _, _ = resize_and_crop(local_cond, image_shape, crop_shape)
                        if task == 'sketch' or task == 'hedsketch':
                            local_cond = 255 - local_cond
                        local_conds.append(local_cond)
                        local_tasks.append(task)
                        local_load_status.append(True)
                    else:
                        local_cond = np.zeros_like(image)
                        local_conds.append(local_cond)
                        local_tasks.append(task)
                        local_load_status.append(False)
            if len(self.local_tasks)>0 and any([cond is None for cond in local_conds]):
                index = (index + 1) % len(self.data)
                continue

            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) / 127.5) - 1.0
        local_conditions = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in local_conds]

        # randomly drop
        rand_num = random.random()
        drop_all_local_prob = self.drop_all_local_prob
        if rand_num < self.drop_all_prob:
            text = ''
            drop_all_local_prob = 1.0
        elif rand_num < self.drop_all_prob + self.drop_txt_prob:
            text = ''

        # drop conditions
        drop_each_cond_prob = [self.drop_each_cond_prob[task] for task in local_tasks]
        local_conditions, local_status = keep_and_drop_with_status(local_conditions, self.keep_all_local_prob, drop_all_local_prob, drop_each_cond_prob)
        local_status = [s1 and s2 for s1, s2 in zip(local_status, local_load_status)]

        # set prompt
        local_prompts = [self.TASK_PROMPT[task] for status, task in zip(local_status, local_tasks) if status]
        c_text = ' an image conditioned on ' + ' and '.join(local_prompts) if len(local_prompts)>0 else ''
        text = text + c_text.strip()
        text = text.strip()

        # to DataContainer
        local_conditions = DataContainer(local_conditions)
        global_conditions = dict(clipembedding=np.zeros((1, 768), dtype=np.float32), color=np.zeros((1,180), dtype=np.float32)) 
        global_conditions = DataContainer(global_conditions)

        return dict(jpg=image, txt=text, local_conditions=local_conditions, global_conditions=global_conditions)
        
    def __len__(self):
        return len(self.data)
