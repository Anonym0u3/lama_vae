import torch
from torch import Tensor, optim
import pytorch_lightning as pl
import torch.nn.functional as F
from model.modules import FFCResNetGenerator_lama,LaMa
from model.help_function import to_torch, resize_square, undo_resize_square, vae_load_checkpoint, disabled_train, freeze_model, freeze_module
from tqdm import trange
from PIL import Image
import numpy as np  
from typing import Any, Dict, Set
from pytorch_lightning.utilities.types import STEP_OUTPUT


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])
    def kl_loss(self, other=None):
        return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / (other.var ** 2)
                    + torch.pow(self.var / other.var, 2) - 1.0 - 2 * self.logvar + 2 * other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class lamaVAE(pl.LightningModule):
    def __init__(self, 
                 lama_ckpt_path,
                 sd_ckpt_path=None,
                 input_nc=4,
                 z_channels=4,
                 lama = False,
                 lr = 0.0001,
                 weight_decay = 1e-6
                 ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = FFCResNetGenerator_lama(input_nc,z_channels)
        self.sd_lama = torch.jit.load(lama_ckpt_path, map_location="cpu").state_dict()
        self.init_from_ckpt(self.sd_lama)
        if lama:
            self.lama = LaMa(self.sd_lama)
            freeze_module(self.lama)
        if sd_ckpt_path is not None:
            self.vae = vae_load_checkpoint(sd_ckpt_path)

    def init_from_ckpt(self,sd):
        state = {
                k.replace("generator.model", "model"): v
                for k, v in sd.items()
            }
        
        self.model.load_state_dict(state,strict=False)
        self.model = freeze_model(self.model,state)
    
    def inpaint(
        self,
        inpaint_model,
        image: Tensor,
        mask: Tensor,
        seed = 666,
    ):
        required_size = 256

        image, mask = to_torch(image, mask)
        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            mask = mask[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
 
        #image_device = image.device
        inpaint_model.to(self.device)
        batch_image = []
        with torch.no_grad():
            #for i in trange(batch_size):
            for i in range(batch_size):
                work_image, work_mask = image[i].unsqueeze(0), mask[i].unsqueeze(0)
                work_image, work_mask, original_size = resize_square(
                    work_image, work_mask, required_size
                )
                work_mask = work_mask.floor()

                torch.manual_seed(seed)
                #work_image = inpaint_model(work_image.to(self.device), work_mask.to(self.device))
                work_image = inpaint_model(work_image, work_mask)

                #work_image.to(image_device)
                work_image = undo_resize_square(work_image, original_size)
                work_image = image[i] + (work_image - image[i]) * mask[i].floor()

                batch_image.append(work_image)

            #inpaint_model.cpu()
            result = torch.cat(batch_image, dim=0)
            result = result.permute(0, 2, 3, 1)  # BCHW -> BHWC
        return result
    
    def forward_lama(self, image, mask):
        result = self.inpaint(
            #seed=random.randint(1, 2**64),
            seed=666,
            inpaint_model=self.lama,
            image=image,
            mask=mask,
        )
        #result_sq = result.squeeze(0)
        #i = 255. * result.cpu().numpy()
        #i = 255. * result_sq.detach().numpy()
        #img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        #img.save('/home/user01/lama_VAE/debug.png')
        #print("done!")
        return result

    def encode(self, image):
        with torch.no_grad():
            latent, z , end_out = self.vae.encode(image)
        return latent, z , end_out
    
    def decode(self, z):
        with torch.no_grad():
            img = self.vae.decode(z)
        return img
    
    def forward(self, image, mask):
        
        return self.model(image, mask)

    def get_loss(self, pred_z, gt_z, latent, pred_end_out, gt_end_out):
        pred_posterior = DiagonalGaussianDistribution(pred_z)
        gt_posterior = DiagonalGaussianDistribution(gt_z)
        pred_latent = pred_posterior.sample()
        mse_loss = F.mse_loss(pred_latent, latent, reduction='mean')
        end_mse_loss = F.mse_loss(pred_end_out, gt_end_out, reduction='mean')
        #kl_loss = pred_posterior.kl_loss(gt_posterior)
        #kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        #kl_loss = torch.clamp(kl_loss, 0, 1)
        #kl_loss = torch.log(kl_loss)
        loss = mse_loss  + end_mse_loss
        return {'mse_loss': mse_loss,'end_loss': end_mse_loss,'loss': loss}
    
    def _common_step(self, batch, batch_idx):
        img = batch['image']
        mask = batch['mask']
        pred_end_out, pred_z = self.model(img, mask)
        mid = self.forward_lama(img, mask)
        latent, gt_z, gt_end_out = self.encode(mid)
        loss = self.get_loss(pred_z, gt_z, latent, pred_end_out, gt_end_out)
        return loss, img, mask, pred_z, latent, gt_z

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        loss, _, _, _, _, _ = self._common_step(batch, batch_idx)
        self.log("mse_loss", loss['mse_loss'])
        self.log("end_loss", loss['end_loss'])
        self.log("train_loss", loss['loss'])
        return loss['loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, img, mask, pred_z, latent, gt_z = self._common_step(batch, batch_idx)
        gt_decode_img = self.decode(latent)
        pred_posterior = DiagonalGaussianDistribution(pred_z)
        pred_latent = pred_posterior.sample()
        pred_decode_img = self.decode(pred_latent)
        self.log("val_loss", loss['loss'])
        return gt_decode_img, pred_decode_img


    def configure_optimizers(self) -> optim.AdamW:
        """
        Configure optimizer for this model.
        
        Returns:
            optimizer (optim.AdamW): The optimizer for this model.
        """
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay= self.weight_decay #筛选出所有需要梯度的参数
        )
        return optimizer
    