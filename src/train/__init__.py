import os
from ldm.util import instantiate_from_config


def create_dataset(cfg):
    cfg.params.local_tasks = cfg.local_tasks
    json_files = [(cfg.json_files[ds], ds) for ds in cfg.datasets]
    cfg.params.json_files = json_files
    return instantiate_from_config(cfg)
