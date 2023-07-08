import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from PIL import Image
import loss
import evaluation as eva
import net
# import export
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms_factory import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
import matplotlib.pyplot as plt

image = Image.open("image.jpeg")
rand_tfm = rand_augment_transform(config_str='rand-mstd1', hparams={'img_mean': (124, 116, 104)})
crop = RandomResizedCropAndInterpolation(size=598, scale=(0.8, 1))
#rand_tfm = create_transform(224, is_training=True, auto_augment='rand-mstd1')
fig, ax = plt.subplots(2, 4, figsize=(10,5))
for idx, im in enumerate([rand_tfm(crop(image)) for i in range(4)]):
    ax[0, idx].imshow(im)
for idx, im in enumerate([rand_tfm(crop(image)) for i in range(4)]):
    ax[1, idx].imshow(im)

fig.tight_layout()
plt.savefig('output.png')
