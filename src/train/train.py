import os
import sys
sys.path.append(os.getcwd())
import argparse
	
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from ldm.util import instantiate_from_config
from models.util import load_state_dict, load_ckpt
from models.logger import ImageLogger
from src.train import create_dataset
from src.train.util import interleaved_collate


parser = argparse.ArgumentParser(description='AnyControl Training')
parser.add_argument('--config-path', type=str, default='./configs/anycontrol_local.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--training-steps', type=int, default=1e5)
parser.add_argument('--resume-path', type=str, default='./ckpts/init_local.ckpt')
parser.add_argument('--logdir', type=str, default='./log_local/')
parser.add_argument('--log-freq', type=int, default=500)
parser.add_argument('--sd-locked', type=bool, default=True)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--gpus', type=int, default=-1)
parser.add_argument('--local-rank', type=int)
args = parser.parse_args()


def main():

    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_steps = args.training_steps
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    dataset = create_dataset(config['data'])
    dataloader = DataLoader(dataset, collate_fn=interleaved_collate, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle=True)

    logger = ImageLogger(
        batch_frequency=logger_freq,
        log_images_kwargs=config['logger'],
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
    )
        
    trainer = pl.Trainer(
        callbacks=[logger, checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
        accelerator='gpu', 
    )
    trainer.fit(model,
        dataloader, 
    )


if __name__ == '__main__':
    main()
