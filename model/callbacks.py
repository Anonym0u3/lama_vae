import torch
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
# or
# from wandb.integration.lightning.fabric import WandbLogger

__all__ = [
    "ModelCheckpoint",
    "LogPredictionSamplesCallback",
    "MyPrintingCallback"
]

class LogPredictionSamplesCallback(Callback):
    def save_img(self,img):
        i = 255. * img.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 4
            img = batch['image']
            mask = batch['mask']
            images = [img for img in img[:n]]
            masks = [mask for mask in mask[:n]]
            gt_decode_img, pred_decode_img = outputs
            gt_decode_imgs = [self.save_img(gt_decode_img) for gt_decode_img in gt_decode_img[:n]]
            pred_decode_imgs = [self.save_img(pred_decode_img) for pred_decode_img in pred_decode_img[:n]]
            outputimgs = gt_decode_imgs + pred_decode_imgs
            captions = ["Ground Truth"]*4 + ["Prediction"]*4

            wandb_logger = WandbLogger(project="lamaVAE", name="train_vae")
            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key="sample_images", images=outputimgs, caption=captions)

            """         
            # Option 2: log images and predictions as a W&B Table
            columns = ["image", "ground truth", "prediction"]
            data = [
                [wandb.Image(x_i), y_i, y_pred] or x_i,
                y_i,
                y_pred in list(zip(x[:n], y[:n], outputs[:n])),
            ]
            wandb_logger.log_table(key="sample_table", columns=columns, data=data)
            """

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print(f"Training is done.")