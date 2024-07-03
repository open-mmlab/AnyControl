import logging

from omegaconf import OmegaConf
from lavis.models import registry
from lavis.models import load_preprocess

from ldm.util import instantiate_from_config


def load_blip2_model(cfg, is_eval=False, device="cpu"):
    model_cls = registry.get_model_class(cfg.model_name)

    # load preprocess
    default_cfg = OmegaConf.load(model_cls.default_config_path(cfg.model_type))
    default_cfg.model.pretrained = cfg.pretrained

    if default_cfg.model.image_size != cfg.params.img_size:
        default_cfg.model.image_size = cfg.params.img_size
    model = model_cls.from_config(default_cfg.model)
    model.cfg = default_cfg.model

    if is_eval:
        model.eval()

    if default_cfg is not None:
        preprocess_cfg = default_cfg.preprocess
        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors


def load_qformer_model(cfg):
    blip2_model, vis_processor, txt_processor = load_blip2_model(cfg) 
    q_former = instantiate_from_config(cfg)
    if blip2_model.query_tokens.shape != q_former.query_tokens.shape:
        blip2_model.query_tokens = q_former.query_tokens
    model_name = cfg.params.get('model_name', 'bert-base-uncased')
    if model_name == 'bert-base-uncased':
        q_former.load_state_dict(blip2_model.state_dict(), strict=False)
    return q_former, (vis_processor, txt_processor)
