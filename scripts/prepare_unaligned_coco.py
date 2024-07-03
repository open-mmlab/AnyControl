import os
import cv2
import sys
import json
import random
import argparse
import gradio as gr
import numpy as np
import torch

from tqdm import tqdm
from copy import deepcopy
from PIL import Image, ImageFilter

from pycocotools.coco import COCO

from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
sys.path.append(os.path.join(os.getcwd(), "data/PowerPaint"))
from pipeline.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from pipeline.pipeline_PowerPaint_ControlNet import StableDiffusionControlNetInpaintPipeline as controlnetPipeline
from utils.utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model
torch.set_grad_enabled(False)

weight_dtype = torch.float16
global pipe

global current_control
current_control = 'canny'

# controlnet_conditioning_scale = 0.8
def resize_image_target(target_image, resolution):
        H, W, C = target_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        img = cv2.resize(target_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

def rle_to_mask(rle) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_depth_map(image):
    image = feature_extractor(
        images=image, return_tensors='pt').pixel_values.to('cuda')
    with torch.no_grad(), torch.autocast('cuda'):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode='bicubic',
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def add_task(prompt, negative_prompt, control_type):
    # print(control_type)
    if control_type == 'object-removal':
        promptA = prompt + ' P_ctxt'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    elif control_type == 'shape-guided':
        promptA = prompt + ' P_shape'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
    elif control_type == 'image-outpainting':
        promptA = prompt + ' P_ctxt'
        promptB = prompt + ' P_ctxt'
        negative_promptA = negative_prompt + ' P_obj'
        negative_promptB = negative_prompt + ' P_obj'
    else:
        promptA = prompt + ' P_obj'
        promptB = prompt + ' P_obj'
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt

    return promptA, promptB, negative_promptA, negative_promptB


def predict(input_image, prompt, fitting_degree, ddim_steps, scale, seed,
            negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio):
    size1, size2 = input_image['image'].convert('RGB').size

    if task!='image-outpainting':
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (640, int(size2 / size1 * 640)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 640), 640))
    else:
        if size1 < size2:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (512, int(size2 / size1 * 512)))
        else:
            input_image['image'] = input_image['image'].convert('RGB').resize(
                (int(size1 / size2 * 512), 512))
        
    if vertical_expansion_ratio!=None and horizontal_expansion_ratio!=None:
        o_W,o_H = input_image['image'].convert('RGB').size
        c_W = int(horizontal_expansion_ratio*o_W)
        c_H = int(vertical_expansion_ratio*o_H)

        expand_img = np.ones((c_H, c_W,3), dtype=np.uint8)*127
        original_img = np.array(input_image['image'])
        expand_img[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W,3), dtype=np.uint8)*255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0):int((c_H-o_H)/2.0)+o_H,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio!=1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0)+blurry_gap:int((c_W-o_W)/2.0)+o_W-blurry_gap,:] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio==1:
            expand_mask[int((c_H-o_H)/2.0)+blurry_gap:int((c_H-o_H)/2.0)+o_H-blurry_gap,int((c_W-o_W)/2.0):int((c_W-o_W)/2.0)+o_W,:] = 0
        
        input_image['image'] = Image.fromarray(expand_img)
        input_image['mask'] = Image.fromarray(expand_mask)

    promptA, promptB, negative_promptA, negative_promptB = add_task(
        prompt, negative_prompt, task)
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image['image'].convert('RGB'))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))
    set_seed(seed)
    global pipe
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps).images[0]
    mask_np = np.array(input_image['mask'].convert('RGB'))
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image['mask'].convert('RGB'), result_m]

    dict_out = [input_image['image'].convert('RGB'), result_paste]

    return dict_out, dict_res


def predict_controlnet(input_image, input_control_image, control_type, prompt,
                       ddim_steps, scale, seed, negative_prompt,controlnet_conditioning_scale):
    promptA = prompt + ' P_obj'
    promptB = prompt + ' P_obj'
    negative_promptA = negative_prompt 
    negative_promptB = negative_prompt 
    size1, size2 = input_image['image'].convert('RGB').size

    if size1 < size2:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (640, int(size2 / size1 * 640)))
    else:
        input_image['image'] = input_image['image'].convert('RGB').resize(
            (int(size1 / size2 * 640), 640))
    img = np.array(input_image['image'].convert('RGB'))
    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image['image'] = input_image['image'].resize((H, W))
    input_image['mask'] = input_image['mask'].resize((H, W))

    global current_control
    global pipe

    base_control = ControlNetModel.from_pretrained(
        'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
    control_pipe = controlnetPipeline(pipe.vae, pipe.text_encoder,
                                      pipe.tokenizer, pipe.unet, base_control,
                                      pipe.scheduler, None, None, False)
    control_pipe = control_pipe.to('cuda')
    current_control = 'canny'
    if current_control != control_type:
        if control_type == 'canny' or control_type is None:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-canny', torch_dtype=weight_dtype)
        elif control_type == 'pose':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-openpose', torch_dtype=weight_dtype)
        elif control_type == 'depth':
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-depth', torch_dtype=weight_dtype)
        else:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                'lllyasviel/sd-controlnet-hed', torch_dtype=weight_dtype)
        control_pipe = control_pipe.to('cuda')
        current_control = control_type

    controlnet_image = input_control_image
    if current_control == 'canny':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = np.array(controlnet_image)
        controlnet_image = cv2.Canny(controlnet_image, 100, 200)
        controlnet_image = controlnet_image[:, :, None]
        controlnet_image = np.concatenate(
            [controlnet_image, controlnet_image, controlnet_image], axis=2)
        controlnet_image = Image.fromarray(controlnet_image)
    elif current_control == 'pose':
        controlnet_image = openpose(controlnet_image)
    elif current_control == 'depth':
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = get_depth_map(controlnet_image)
    else:
        controlnet_image = hed(controlnet_image)

    mask_np = np.array(input_image['mask'].convert('RGB'))
    controlnet_image = controlnet_image.resize((H, W))
    set_seed(seed)
    result = control_pipe(
        promptA=promptB,
        promptB=promptA,
        tradoff=1.0,
        tradoff_nag=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image['image'].convert('RGB'),
        mask_image=input_image['mask'].convert('RGB'),
        control_image=controlnet_image,
        width=H,
        height=W,
        guidance_scale=scale,
        controlnet_conditioning_scale = controlnet_conditioning_scale,
        num_inference_steps=ddim_steps).images[0]
    red = np.array(result).astype('float') * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (result_m.astype('float') * (1 - mask_np.astype('float') / 512.0) +
         mask_np.astype('float') / 512.0 * red).astype('uint8'))

    mask_np = np.array(input_image['mask'].convert('RGB'))
    m_img = input_image['mask'].convert('RGB').filter(
        ImageFilter.GaussianBlur(radius=4))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image['image'].convert('RGB')) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))
    return [input_image['image'].convert('RGB'), result_paste], [controlnet_image, result_m]


def infer(input_image, text_guided_prompt='', text_guided_negative_prompt='',
          shape_guided_prompt='', shape_guided_negative_prompt='', fitting_degree=1,
          ddim_steps=20, scale=15, seed=1362348480, task='object-removal', enable_control=True, input_control_image=None,
          control_type=None,vertical_expansion_ratio=None,horizontal_expansion_ratio=None,
          outpaint_prompt=None,outpaint_negative_prompt=None,controlnet_conditioning_scale=None,
          removal_prompt='',removal_negative_prompt=''):
    if task == 'text-guided':
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    elif task == 'shape-guided':
        prompt = shape_guided_prompt
        negative_prompt = shape_guided_negative_prompt
    elif task == 'object-removal':
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == 'image-outpainting':
        prompt = outpaint_prompt
        negative_prompt = outpaint_negative_prompt
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task,vertical_expansion_ratio,horizontal_expansion_ratio)
    else:
        task = 'text-guided'
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    if enable_control and task == 'text-guided':
        return predict_controlnet(input_image, input_control_image,
                                  control_type, prompt, ddim_steps, scale,
                                  seed, negative_prompt,controlnet_conditioning_scale)
    else:
        return predict(input_image, prompt, fitting_degree, ddim_steps, scale,
                       seed, negative_prompt, task,None,None)


def select_tab_text_guided():
    return 'text-guided'


def select_tab_object_removal():
    return 'object-removal'

def select_tab_image_outpainting():
    return 'image-outpainting'


def select_tab_shape_guided():
    return 'shape-guided'


def unique_id():
    cands = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    sid = ''.join(random.choices(cands, k=8))
    return sid


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AnyControl Training Data')
    parser.add_argument('--model_dir', type=str, default="./data")
    parser.add_argument('--data_root', type=str, default="./datasets")
    args = parser.parse_args()

    # define models
    pp_path = os.path.join(args.model_dir, "PowerPaint-v1")
    assert os.path.exists(pp_path), \
        f"File {pp_path} does not exist. You must download PowerPaint-v1 model from https://huggingface.co/JunhaoZhuang/PowerPaint-v1 first."

    pipe = Pipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=weight_dtype)
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained='runwayml/stable-diffusion-v1-5',
        subfolder='tokenizer',
        revision=None)
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=['P_ctxt', 'P_shape', 'P_obj'],
        initialize_tokens=['a', 'a', 'a'],
        num_vectors_per_token=10)
    load_model(pipe.unet, f"{pp_path}/unet/unet.safetensors")
    load_model(pipe.text_encoder, f"{pp_path}/text_encoder/text_encoder.safetensors")
    pipe = pipe.to('cuda')
    
    # generate
    save_json = os.path.join(args.data_root, "MSCOCO/annotations/instances_inpaint_train2017.json")
    output = os.path.join(args.data_root, "MSCOCO/conditions/inpaint/")
    mask_output = os.path.join(args.data_root, "MSCOCO/conditions/mask/")

    img_source = os.path.join(args.data_root, 'MSCOCO/train2017/')
    annotation_file = os.path.join(args.data_root, "MSCOCO/annotations/instances_train2017.json") 
    caption_file = os.path.join(args.data_root, "MSCOCO/annotations/captions_train2017.json")

    captions = json.load(open(caption_file))['annotations']
    captions = {cap['image_id']:cap['caption'] for cap in captions}
    os.makedirs(output, exist_ok=True)
    os.makedirs(mask_output, exist_ok=True)

    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = sorted(coco.getImgIds())
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    writer = open(save_json, 'w')

    for i,imgId in enumerate(tqdm(imgIds, ncols=100)):
        img = coco.loadImgs(imgId)[0]
        caption = captions[imgId]
        img_path = os.path.join(img_source, img['file_name'])
        image = Image.open(img_path)
        w, h = image.size
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for i in range(len(anns)):
            ratio=anns[i]['area']/(h*w)
            if ratio<0.1 or ratio>0.4:
                continue

            mask = coco.annToMask(anns[i])
            maskf = deepcopy(mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(maskf, kernel, iterations = 10)
            mask = mask.astype(np.uint8)*255
            mask = Image.fromarray(mask)

            input_image={'image':image,'mask':mask}
            inpaint_result, gallery = infer(input_image=input_image)

            cat = coco.loadCats([anns[i]['category_id']])[0]
            cat = cat['name']

            obj_id = anns[i]['id']
            uid = '%012d' % obj_id
            mask_id = img['file_name'][:-4] + '_' + uid
            output_path = os.path.join(output, mask_id+'.jpg')
            mask_output_path = os.path.join(mask_output, mask_id+'.jpg')

            ret = cv2.resize(np.array(inpaint_result[1]), (w, h))
            saved_mask = cv2.resize((maskf*255).astype(np.uint8), (w, h))
            Image.fromarray(ret).save(output_path)
            Image.fromarray(saved_mask).save(mask_output_path)
            
            save_info = {
                'image_id': img['file_name'],
                'class': cat,
                'mask': mask_id,
                'inpaint': mask_id,
                'prompt': caption,
            }
            writer.write(json.dumps(save_info)+'\n')
