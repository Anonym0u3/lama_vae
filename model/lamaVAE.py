import torch
import logging
from torch import Tensor, optim, nn
import pytorch_lightning as pl
from model.modules import FFCResNetGenerator_lama,LaMa
from model.help_function import to_torch, resize_square, undo_resize_square, vae_load_checkpoint, default, freeze_model, freeze_module, instantiate_from_config
import numpy as np  
from typing import Any, Dict, List, Tuple

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

class DiagonalGaussianRegularizer(nn.Module):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log

class lamaVAE(pl.LightningModule):
    def __init__(self, 
                 lama_ckpt_path,
                 loss_config,
                 sd_ckpt_path=None,
                 input_nc=4,
                 z_channels=4,
                 lama = False,
                 lr = 0.0001,
                 weight_decay = 1e-6,
                 disc_start_iter = 0
                 ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = FFCResNetGenerator_lama(input_nc,z_channels)
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        self.sd_lama = torch.jit.load(lama_ckpt_path, map_location="cpu").state_dict()
        self.init_from_ckpt(self.sd_lama)
        self.disc_start_iter = disc_start_iter
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
    
    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params
    
    def get_last_layer(self):
        return self.vae.first_stage_model.decoder.get_last_layer()
    
    def encode_vae(self, image):
        with torch.no_grad():
            latent, z = self.vae.encode(image)
        return latent, z
    
    def encode(self, image, mask, return_reg_log=False,unregularized=False):
        z = self.model(image, mask)
        if unregularized:
            return z, dict()
        z, reg_log = DiagonalGaussianRegularizer(z)
        if return_reg_log:
            return z, reg_log
        return z
    
    def decode(self, z):
        img = self.vae.decode(z)
        return img
    
    def forward(self, image, mask):
        z, reg_log = self.encode(image, mask, return_reg_log=True)
        decode_img = self.decode(z)
        return z , decode_img, reg_log

    def inner_training_step(
        self, batch: dict, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        img = batch['image']
        mask = batch['mask']

        z, xrec, regularization_log = self(img, mask)
        gt_lama = self.forward_lama(img, mask)

        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(gt_lama, xrec, **extra_info)

            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(gt_lama, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")
    
    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(
                batch, batch_idx, optimizer_idx=optimizer_idx
            )
            self.manual_backward(loss)
        opt.step()        

    def validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        img = batch['image']
        mask = batch['mask']

        z, xrec, regularization_log = self(img, mask)
        gt_lama = self.forward_lama(img, mask)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(gt_lama, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(gt_lama, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return gt_lama, xrec
    
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        ae_params = [p for p in self.model.parameters() if p.requires_grad]

        disc_params = self.get_discriminator_params()
        opt_ae = optim.AdamW(
            ae_params, lr=default(self.lr_g_factor, 1.0) * self.lr, weight_decay= self.weight_decay #筛选出所有需要梯度的参数
        )

        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = optim.AdamW(
            disc_params, lr=self.lr, weight_decay= self.weight_decay #筛选出所有需要梯度的参数
        )
            opts.append(opt_disc)

        return opts