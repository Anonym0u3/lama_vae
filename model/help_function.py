import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
import numpy as np 
import comfy.utils
from comfy import model_detection
from model.modules import VAE
from typing import Mapping, Any
import importlib
from torch import nn

def to_torch(image, mask):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        if len(mask.shape) == 3:  # BHW -> B1HW
            mask = mask.unsqueeze(1)
        elif len(mask.shape) == 2:  # HW -> B1HW
            mask = mask.unsqueeze(0).unsqueeze(0)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask

def resize_square(image, mask, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="reflect")

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)

def undo_resize_square(image, original_size):  # original_size: tuple[int, int, int]
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]

def load_image(image_path):
    img = Image.open(image_path)
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image

def load_maskimage(image_path, channel):
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    if i.getbands() != ("R", "G", "B", "A"):
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        i = i.convert("RGBA")
    mask = None
    c = channel[0].upper()
    if c in i.getbands():
        mask = np.array(i.getchannel(c)).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        if c == 'A':
            mask = 1. - mask
    else:
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
    return mask.unsqueeze(0)


def vae_load_checkpoint(ckpt_path,dtype=torch.float32):
    sd = comfy.utils.load_torch_file(ckpt_path)
    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.")
    
    vae_sd = comfy.utils.state_dict_prefix_replace(sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True)
    vae_sd = model_config.process_vae_state_dict(vae_sd)
    #vae = VAE(sd=vae_sd)
    vae = VAE(sd=vae_sd, dtype=dtype)
    return vae

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def freeze_model(model, to_freeze_dict, keep_step=None):

    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    # # 打印当前的固定情况（可忽略）：
    # freezed_num, pass_num = 0, 0
    # for (name, param) in model.named_parameters():
    #     if param.requires_grad == False:
    #         freezed_num += 1
    #     else:
    #         pass_num += 1
    # print('\n Total {} params, miss {} \n'.format(freezed_num + pass_num, pass_num))

    return model

def get_obj_from_str(string: str, reload: bool=False) -> object: # 根据给定的str动态地加载模块和类
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config: Mapping[str, Any]) -> object: # 根据给定的配置文件动态地创建并初始化对象
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def freeze_module(module: nn.Module) -> None:
    module.eval() #设为eval模式
    module.train = disabled_train #禁用模型的train模式
    for p in module.parameters(): #冻结模型的参数
        p.requires_grad = False

def load_state_dict(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)