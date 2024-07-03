import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def load_ckpt(model, state_dict, strict=True, input_channel_copy_indices=None):
    input_channel_key = "local_adapter.feature_extractor.pre_extractor.0.weight"
    model_state_dict = model.state_dict()
    if input_channel_key in model_state_dict and input_channel_key in state_dict:
        model_shape = model_state_dict[input_channel_key].shape
        shape = state_dict[input_channel_key].shape
        if model_shape != shape:
            if input_channel_copy_indices is None:
                state_dict[input_channel_key] = state_dict[input_channel_key][:, :model_shape[1], :, :]
            else:
                cout, cin, h, w = model_shape
                weight = state_dict[input_channel_key].view(cout, -1, cin//3, h, w)
                weight = weight[:, input_channel_copy_indices].view(cout, cin, h, w)
                state_dict[input_channel_key] = weight
    model.load_state_dict(state_dict, strict=strict)


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
