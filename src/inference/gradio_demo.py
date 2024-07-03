import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "annotator/entityseg"))
import cv2
import einops
import torch
import gradio as gr
import numpy as np
from pytorch_lightning import seed_everything
from PIL import Image

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.entityseg import EntitysegDetector
from annotator.openpose import OpenposeDetector
from annotator.content import ContentDetector
from annotator.cielab import CIELabDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

'''
define conditions
'''
max_conditions = 8
condition_types = ["edge", "depth", "seg", "pose", "content", "color"]

apply_canny = CannyDetector()
apply_midas = MidasDetector()
apply_seg = EntitysegDetector()
apply_openpose = OpenposeDetector()
apply_content = ContentDetector()
apply_color = CIELabDetector()

processors = {
    "edge": apply_canny,
    "depth": apply_midas,
    "seg": apply_seg,
    "pose": apply_openpose,
    "content": apply_content,
    "color": apply_color,
}

descriptors = {
    "edge": "canny",
    "depth": "depth",
    "seg": "segmentation",
    "pose": "openpose",
}


@torch.no_grad()
def get_unconditional_global(c_global):
    if isinstance(c_global, dict):
        return {k:torch.zeros_like(v) for k,v in c_global.items()}
    elif isinstance(c_global, list):
        return [torch.zeros_like(c) for c in c_global]
    else:
        return torch.zeros_like(c_global) 


def process(prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, 
            strength, scale, seed, eta, global_strength, color_strength, local_strength, *args):
    
    seed_everything(seed)

    conds_and_types = args
    conds = conds_and_types[0::2]
    types = conds_and_types[1::2]
    conds = [c for c in conds if c is not None]
    types = [t for t in types if t is not None]
    assert len(conds) == len(types)

    detected_maps = []
    other_maps = []
    tasks = []

    # initialize global control
    global_conditions = dict(clipembedding=np.zeros((1, 768), dtype=np.float32), color=np.zeros((1, 180), dtype=np.float32))
    global_control = {}
    for key in global_conditions.keys():
        global_cond = torch.from_numpy(global_conditions[key]).unsqueeze(0).repeat(num_samples, 1, 1)
        global_cond = global_cond.cuda().to(memory_format=torch.contiguous_format).float()
        global_control[key] = global_cond

    # initialize local control
    anchor_image = HWC3(np.zeros((image_resolution, image_resolution, 3)).astype(np.uint8))
    oH, oW = anchor_image.shape[:2]
    H, W, C = resize_image(anchor_image, image_resolution).shape
    anchor_tensor = ddim_sampler.model.qformer_vis_processor['eval'](Image.fromarray(anchor_image))
    local_control = torch.tensor(anchor_tensor).cuda().to(memory_format=torch.contiguous_format).half()

    task_prompt = ''

    with torch.no_grad():

        # set up local control
        for cond, typ in zip(conds, types):
            if typ in ['edge', 'depth', 'seg', 'pose']:
                oH, oW = cond.shape[:2] 
                cond_image = HWC3(cv2.resize(cond, (W, H)))
                cond_detected_map = processors[typ](cond_image)
                cond_detected_map = HWC3(cond_detected_map)
                detected_maps.append(cond_detected_map)
                tasks.append(descriptors[typ])
            elif typ in ['content']:
                other_maps.append(cond)
                content_image = cv2.cvtColor(cond, cv2.COLOR_RGB2BGR)
                content_emb = apply_content(content_image)
                global_conditions['clipembedding'] = content_emb
            elif typ in ['color']:
                color_hist = apply_color(cond)
                global_conditions['color'] = color_hist
                color_palette = apply_color.hist_to_palette(color_hist) # (50, 189, 3)
                color_palette = cv2.resize(color_palette, (W, H), cv2.INTER_NEAREST)
                other_maps.append(color_palette)
        if len(detected_maps) > 0:
            local_control = torch.cat([ddim_sampler.model.qformer_vis_processor['eval'](Image.fromarray(img)).cuda().unsqueeze(0) for img in detected_maps], dim=1)
            task_prompt = ' conditioned on ' + ' and '.join(tasks)
        local_control = local_control.repeat(num_samples, 1, 1, 1)

        # set up global control
        for key in global_conditions.keys():
            global_cond = torch.from_numpy(global_conditions[key]).unsqueeze(0).repeat(num_samples, 1, 1)
            global_cond = global_cond.cuda().to(memory_format=torch.contiguous_format).float()
            global_control[key] = global_cond

        # set up prompt
        input_prompt = (prompt + ' ' + task_prompt).strip()

        # set up cfg
        uc_local_control = local_control
        uc_global_control = get_unconditional_global(global_control)
        cond = {
            "local_control": [local_control], 
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 
            "global_control": [global_control],
            "text": [[input_prompt] * num_samples],
        }
        un_cond = {
            "local_control": [uc_local_control], 
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 
            'global_control': [uc_global_control],
            "text": [[input_prompt] * num_samples],
        }
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond, 
                                                     global_strength=global_strength, 
                                                     color_strength=color_strength,
                                                     local_strength=local_strength)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]

    results = [cv2.resize(res, (oW, oH)) for res in results]
    detected_maps = [cv2.resize(maps, (oW, oH)) for maps in detected_maps]
    return [results, detected_maps+other_maps]

  
def variable_image_outputs(k):
    if k is None:
        k = 1
    k = int(k)
    imageboxes = []
    for i in range(max_conditions):
        if i<k:
            with gr.Row(visible=True):
                img = gr.Image(sources=['upload'], type="numpy", label=f'Condition {i+1}', visible=True, interactive=True, scale=3, height=200)
                typ = gr.Dropdown(condition_types, visible=True, interactive=True, label="type", scale=1)
        else:
            with gr.Row(visible=False):
                img = gr.Image(sources=['upload'], type="numpy", label=f'Condition {i+1}', visible=False, scale=3, height=200)
                typ = gr.Dropdown(condition_types, visible=False, interactive=True, label="type", scale=1)
        imageboxes.append(img)
        imageboxes.append(typ)
    return imageboxes


'''
define model
'''
config_file = "configs/anycontrol.yaml"
model_file = "ckpts/anycontrol_15.ckpt"
model = create_model(config_file).cpu()
model.load_state_dict(load_state_dict(model_file, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



block = gr.Blocks(theme='bethecloud/storj_theme').queue()
with block:
    with gr.Row():
        gr.Markdown("## AnyControl Demo")
        gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Blocks():
                s = gr.Slider(1, max_conditions, value=1, step=1, label="How many conditions to upload:")
                imageboxes = []
                for i in range(max_conditions):
                    if i==0:
                        with gr.Row():
                            img = gr.Image(visible=True, sources=['upload'], type="numpy", label='Condition 1', interactive=True, scale=3, height=200)
                            typ = gr.Dropdown(condition_types, visible=True, interactive=True, label="type", scale=1)
                    else:
                        with gr.Row():
                            img = gr.Image(visible=False, sources=['upload'], type="numpy", label=f'Condition {i+1}', scale=3, height=200)
                            typ = gr.Dropdown(condition_types, visible=False, interactive=True, label="type", scale=1)
                    imageboxes.append(img)
                    imageboxes.append(typ)
                s.change(variable_image_outputs, s, imageboxes)
        with gr.Column(scale=2):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt")
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1, step=0.01)

                        local_strength = gr.Slider(label="Local Strength", minimum=0, maximum=2, value=1, step=0.01)
                        global_strength = gr.Slider(label="Global Strength", minimum=0, maximum=2, value=1, step=0.01)
                        color_strength = gr.Slider(label="Color Strength", minimum=0, maximum=2, value=1, step=0.01)
                        
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, value=42, step=1)
                        eta = gr.Number(label="Eta (DDIM)", value=0.0)
                        
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                        n_prompt = gr.Textbox(label="Negative Prompt",
                                              value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')


            with gr.Row():
                run_button = gr.Button(value="Run")
            with gr.Row():
                image_gallery = gr.Gallery(label='Generation', show_label=True, elem_id="gallery", columns=[4], rows=[1], height='auto', interactive=False)
            with gr.Row():
                cond_gallery = gr.Gallery(label='Condition', show_label=True, elem_id="gallery", columns=[4], rows=[1], height='auto', interactive=False)

            inputs = [prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, 
                      strength, scale, seed, eta, local_strength, global_strength, color_strength] + imageboxes
            run_button.click(fn=process, inputs=inputs, outputs=[image_gallery, cond_gallery])


# uncomment this block in case you need it
# os.environ['http_proxy'] = ''
# os.environ['https_proxy'] = ''
# os.environ['no_proxy'] = 'localhost,127.0.0.0/8,127.0.1.1'
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''
# os.environ['NO_PROXY'] = 'localhost,127.0.0.0/8,127.0.1.1'
# os.environ['TMPDIR'] = './tmpfiles'


block.launch(server_name='0.0.0.0', allowed_paths=["."], share=False)
