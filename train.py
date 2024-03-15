from argparse import ArgumentParser
from re import L

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from model.lamaVAE import lamaVAE
from model.help_function import load_image, load_maskimage, load_state_dict, instantiate_from_config
from pytorch_lightning.loggers import WandbLogger
import wandb
def main_0() -> None:
    parser = ArgumentParser()
    #parser.add_argument("--config_lama", type=str, required=True)
    #args = parser.parse_args()
    #config_lama = "/home/user01/lama_VAE/configs/models/lama.yaml"
    # OmegaConf是一个用于处理配置的Python库，它可以将YAML或JSON格式的配置文件转换为Python的字典或列表
    #config_lama = OmegaConf.load(config_lama) # 加载配置文件，并将其转换为一个DictConfig或ListConfig对象
    #pl.seed_everything(config.lightning.seed, workers=True)
    model_file = "/home/user01/ComfyUI/models/inpaint/big-lama.pt"
    sd_ckpt_path = "/home/user01/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors"
    model = lamaVAE(model_file,lama=True,sd_ckpt_path=sd_ckpt_path)
    #print(list(model.state_dict().items())[:5])
    image_path = "/home/user01/lama/output/8399166846_f6fb4e4b8e_k_lamaed.png"
    mask_image_path = "/home/user01/lama/output/8399166846_f6fb4e4b8e_k_mask.png"
    image = load_image(image_path)
    mask = load_maskimage(mask_image_path, "red")
    img_lama = model.forward_lama(image, mask)
    latent, z = model.encode(img_lama)
    #torch.save(latent, '/home/user01/ComfyUI/latent.pt')
    print(latent.size())
    print(z.size())

def main() -> None:
    #parser = ArgumentParser()
    #parser.add_argument("--config", type=str, required=True)
    #args = parser.parse_args()
    #config = OmegaConf.load(args.config)
    wandb.finish()
    wandb_logger = WandbLogger(project="lamaVAE", name="train_vae")

    config = OmegaConf.load("/hy-tmp/lamavae/configs/train_lamavae.yaml") # 加载配置文件，并将其转换为一个DictConfig或ListConfig对象
    pl.seed_everything(config.lightning.seed, workers=True)

    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=True)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))

    trainer = pl.Trainer(callbacks=callbacks,**config.lightning.trainer,logger=wandb_logger)
    trainer.fit(model, datamodule=data_module)
if __name__ == "__main__":
    main()

