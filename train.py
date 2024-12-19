import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        # initializes the NeRF system with hyperparameters, embeddings, and model configurations.
        super().__init__()
        self.save_hyperparameters(vars(hparams))

        self.validation_outputs = []
        self.loss = loss_dict['nerfw'](coef=1)

        # Embedding and model setup code...
    
    def get_progress_bar_dict(self):
        # customizes the progress bar display dictionary, omitting version numbers.
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, outfit_code):
        # performs batched inference on input rays using the NeRF models and embeddings.
        # returns rendered outputs including RGB values and other data.
        B = rays.shape[0]
        results = defaultdict(list)

        # Forward pass and rendering code...

        return results

    def setup(self, stage):
        # sets up the datasets for training and validation based on hyperparameters.
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}

        # Additional dataset-specific arguments...

        self.train_dataset = dataset(split='train', img_wh=tuple(self.hparams.img_wh), **kwargs)
        self.val_dataset = dataset(split='val', img_wh=tuple(self.hparams.img_wh), **kwargs)

    def configure_optimizers(self):
        # configures the optimizer and learning rate scheduler for training the models.
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        # creates and returns the training data loader.
        print(f"Training Dataset Length: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        # creates and returns the validation data loader.
        print(f"Validation Dataset Length: {len(self.val_dataset)}")
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        # defines the training step, calculating losses, logging metrics, and optimizing the models.
        rays, rgbs, ts, outfit_code = batch['rays'], batch['rgbs'], batch['ts'], batch['outfit_code']
        results = self(rays, ts, outfit_code)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        # Logs training metrics...

        return loss

    def validation_step(self, batch, batch_nb):
        # defines the validation step, calculating losses and logging validation metrics.
        # optionally logs sample images and depth maps for visualization.
        rays, rgbs, ts, outfit_code = batch['rays'], batch['rgbs'], batch['ts'], batch['outfit_code']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        results = self(rays, ts, outfit_code)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}

        # Validation visualization code...

        return loss

    def on_validation_epoch_end(self):
        # aggregates and logs validation metrics at the end of each epoch.
        mean_loss = torch.stack([x['val_loss'] for x in self.validation_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in self.validation_outputs]).mean()

        # Logs aggregated metrics and clears validation outputs.

        self.validation_outputs.clear()


def main(hparams):
    # sets up and trains the NeRF system with the specified hyperparameters.
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', hparams.exp_name),
        filename='{epoch:d}',
        monitor='val/psnr',
        mode='max',
        save_top_k=-1
    )

    logger = TensorBoardLogger(
        save_dir='logs',
        name=hparams.exp_name
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        devices=hparams.num_gpus if hparams.num_gpus > 0 else 1,
        accelerator='gpu' if hparams.num_gpus > 0 else 'cpu',
        strategy='ddp' if hparams.num_gpus > 1 else 'auto',
        num_sanity_val_steps=1,
    )

    print("Starting training...")
    trainer.fit(system)


if __name__ == '__main__':
    # entry point for training the NeRF system.
    hparams = get_opts()
    print(hparams)
    main(hparams)
