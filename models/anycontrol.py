import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from models.q_formers import load_qformer_model


class AnyControlNet(LatentDiffusion):

    def __init__(self, mode, qformer_config=None, local_control_config=None, global_control_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['local', 'uni']
        self.mode = mode
        self.qformer_config = qformer_config
        self.local_control_config = local_control_config
        self.global_control_config = global_control_config

        self.model.diffusion_model.requires_grad_(False)
        self.model.diffusion_model.requires_grad_(False)
        self.model.diffusion_model.requires_grad_(False)

        q_former, (vis_processor, txt_processor) = load_qformer_model(qformer_config)
        self.q_former = q_former
        self.qformer_vis_processor = vis_processor
        self.qformer_txt_processor = txt_processor

        self.local_adapter = instantiate_from_config(local_control_config)
        self.local_control_scales = [1.0] * 13
        self.global_adapter = instantiate_from_config(global_control_config) if self.mode == 'uni' else None
        self.clip_embeddings_dim = global_control_config.params.clip_embeddings_dim
        self.color_in_dim = global_control_config.params.color_in_dim

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # latent and text 
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        bs = bs or x.size(0)
        shape = self.get_shape(batch, bs)
        
        local_control = self.get_local_conditions_for_vision_encoder(batch, bs)
        local_control = local_control.to(memory_format=torch.contiguous_format).float()

        global_control = {}
        global_conditions = batch['global_conditions'][:bs]
        for key in batch['global_conditions'][0].data.keys():
            global_cond = torch.stack([torch.Tensor(dc.data[key]) for dc in global_conditions])
            global_cond = global_cond.to(self.device).to(memory_format=torch.contiguous_format).float()
            global_control[key] = global_cond

        conditions = dict(
            text=[batch['txt']], 
            c_crossattn=[c], 
            local_control=[local_control], 
            global_control=[global_control],
        ) 
        return x, conditions 

    def apply_model(self, x_noisy, t, cond, local_strength=1.0, content_strength=1.0, color_strength=1.0, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        text = cond['text'][0]
        bs = x_noisy.shape[0]

        # extract global control
        if self.mode in ['uni']:
            content_control, color_control = self.global_adapter(
                cond['global_control'][0]['clipembedding'], cond['global_control'][0]['color'])
        else:
            content_control = torch.zeros(bs, self.clip_embeddings_dim).to(self.device).to(memory_format=torch.contiguous_format).float()
            color_control = torch.zeros(bs, self.color_in_dim).to(self.device).to(memory_format=torch.contiguous_format).float()

        # extract local control
        if self.mode in ['local', 'uni']:
            local_features = self.local_adapter.extract_local_features(self.q_former, text, cond['local_control'][0])
            local_control = self.local_adapter(x=x_noisy, timesteps=t, context=cond_txt, local_features=local_features)
            local_control = [c * scale for c, scale in zip(local_control, self.local_control_scales)]

        eps = diffusion_model(
            x=x_noisy, timesteps=t, context=cond_txt, 
            local_control=local_control, local_w=local_strength,
            content_control=content_control, extra_w=content_strength, 
            color_control=color_control, color_w=color_strength)
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def get_unconditional_global_conditioning(self, c):
        if isinstance(c, dict):
            return {k:torch.zeros_like(v) for k,v in c.items()}
        elif isinstance(c, list):
            return [torch.zeros_like(v) for v in c]
        else:
            return torch.zeros_like(c) 

    @torch.no_grad()
    def get_shape(self, batch, N):
        return [dc.data[0].shape[:2] for dc in batch['local_conditions'][:N]]

    @torch.no_grad()
    def get_local_conditions_for_vision_encoder(self, batch, N):
        # return: local_conditions, (bs, num_conds * 3, h, w)
        local_conditions = []
        max_len = max([len(dc.data) for dc in batch['local_conditions'][:N]])
        for dc in batch['local_conditions'][:N]:
            conds = torch.cat([self.qformer_vis_processor['eval'](Image.fromarray(img)).unsqueeze(0) for img in dc.data], dim=1)
            local_conditions.append(conds)
        local_conditions = [F.pad(cond, (0,0,0,0,0,max_len*3-cond.shape[1],0,0)) for cond in local_conditions]
        local_conditions = torch.cat(local_conditions, dim=0).to(self.device) 
        return local_conditions

    @torch.no_grad()
    def get_local_conditions_for_logging(self, batch, N):
        local_conditions = []
        max_len = max([len(dc.data) for dc in batch['local_conditions'][:N]])
        for dc in batch['local_conditions'][:N]:
            conds = torch.stack([torch.Tensor(img).permute(2,0,1) for img in dc.data], dim=0) # (n, c, h, w)
            conds = conds.float() / 255.
            conds = conds * 2.0 - 1.0
            local_conditions.append(conds)
        local_conditions = [F.pad(cond, (0,0,0,0,0,0,0,max_len-cond.shape[0])) for cond in local_conditions]
        local_conditions = torch.stack(local_conditions, dim=0).to(self.device) # (bs, n, c, h, w)
        local_conditions = local_conditions.flatten(1,2)
        return local_conditions

    def clip_batch(self, batch, key, N, flag=True):
        if isinstance(batch, torch.Tensor):
            return batch[:N] 
        elif isinstance(batch, list):
            return batch[:N] 
        batch = batch[key][0] if flag else batch[key]
        if isinstance(batch, torch.Tensor):
            return batch[:N] 
        elif isinstance(batch, list):
            return batch[:N] 
        elif isinstance(batch, dict):
            return {k:self.clip_batch(v,'',N,flag=False) for k,v in batch.items()}
        else:
            raise ValueError(f'Unsupported type {type(batch)}')

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, 
                   plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)

        shape = self.get_shape(batch, N)
        c_local = self.clip_batch(c, "local_control", N)
        c_global = self.clip_batch(c, "global_control", N)
        c_context = self.clip_batch(c, "c_crossattn", N)
        c_text = self.clip_batch(batch, self.cond_stage_key, N, False)
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["conditioning"] = log_txt_as_img((512, 512), c_text, size=16)
        log["local_control"] = self.get_local_conditions_for_logging(batch, N)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        cond_dict = dict(
            local_control=[c_local],
            global_control=[c_global],
            c_crossattn=[c_context],
            text=[c_text],
            shape=[shape],
        )

        if sample:
            samples, z_denoise_row = self.sample_log(cond=cond_dict,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta,
                                                     log_every_t=self.log_every_t * 0.05)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                if isinstance(z_denoise_row, dict):
                    for key in ['pred_x0', 'x_inter']:
                       z_denoise_row_key = z_denoise_row[key]
                       denoise_grid = self._get_denoise_row_from_list(z_denoise_row_key)
                       log[f"denoise_row_{key}"] = denoise_grid
                else:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                    log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_context = self.get_unconditional_conditioning(N)
            uc_global = self.get_unconditional_global_conditioning(c_global)
            uc_local = c_local 
            uc_text = c_text 

            uncond_dict = dict(
                local_control=[uc_local],
                global_control=[uc_global],
                c_crossattn=[uc_context],
                text=[uc_text],
                shape=[shape]
            )

            samples_cfg, _ = self.sample_log(cond=cond_dict,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uncond_dict,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if cond['shape'] is None:
            h, w = 512, 512
        else:
            h, w = cond["shape"][0][0]
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.q_former.parameters()) + list(self.local_adapter.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.local_adapter = self.local_adapter.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.local_adapter = self.local_adapter.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
