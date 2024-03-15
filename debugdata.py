from lama_VAE.dataset.lamavae import InpaintingTrainDataset
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from model.help_function import instantiate_from_config

def save_img(path,img):
    #i = 255. * img.detach().numpy()
    img = np.transpose(img, (1 , 2, 0))
    i = 255. * img
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8),mode = 'RGB')
    img.save(path)
def save_img_mask(path,img):
    #i = 255. * img.detach().numpy()
    #img = np.transpose(img, (1 , 2, 0))
    i = 255. * img
    i = i[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8),mode = 'L')
    img.save(path)

config = OmegaConf.load("/home/user01/lama_VAE/configs/dataset/lamavae_train.yaml") # 加载配置文件，并将其转换为一个DictConfig或ListConfig对象

dataset = instantiate_from_config(config.dataset)

print(len(dataset))
image = dataset[1]['image']
mask = dataset[1]['mask']
print(image.size())
print(mask.size())
#save_img('/home/user01/lama_VAE/img.png', image)
#save_img_mask('/home/user01/lama_VAE/mask.png', mask)

"""
mask = dataset[0]['mask']
mask_path = '/home/user01/lama_VAE/mask.png'
cv2.imwrite(mask_path, mask) 
"""
