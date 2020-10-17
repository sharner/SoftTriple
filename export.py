import argparse, sys, os, io, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from PIL import Image
from torchvision import transforms
from torch.utils import data

class DirectoryDataSet(data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self) -> int:
        return len(self.total_imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx]

def RGB2BGR(im):
    assert(im.mode == 'RGB')
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))
        
trans = transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            transforms.Normalize(mean=[104., 117., 128.],
                                        std=[1., 1., 1.]),
        ])

def load_model(model_checkpoint: str) -> nn.Module:
    model = torch.load(model_checkpoint)
    model.cuda(0)
    model.eval()
    return model

def make_loader(image_dir: str, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    return data.DataLoader(
        DirectoryDataSet(image_dir, trans),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

def save_torch(model, base_name):
    save_pytorch(model)
    print("Saving model!")
    fn = "{}.pth".format(base_name)
    torch.save(model, fn)
    print("Model saved to", fn)

def make_script(model: nn.Module, loader: torch.utils.data.DataLoader) -> None:
    labels = []
    with torch.no_grad():
        for i, (img, fn) in enumerate(loader):
            inputs = img.cuda(0)
            traced_script_module = torch.jit.trace(model.module, inputs)
            outputs = traced_script_module(inputs)
            traced_script_module.save("traced_model.pts")
            break

def onnx_export(test_loader, model):
    # switch to evaluation mode
    labels = []
    with torch.no_grad():
        for i, (img, fn) in enumerate(loader):
            inputs = img.cuda(0)
            torch.onnx.export(model.module, input, "model_n1.onnx",
                              input_names=['input'], output_names=['output'],
                              export_params=True)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a SoftTriple model')
    parser.add_argument('--model', type=str,  help='Pickle file with saved model')
    parser.add_argument('--image_dir', type=str, help='directory of images')
    args = parser.parse_args()
    print('Model', args.model)
    print('Image Directory', args.image_dir)
    model = load_model(args.model)
    loader = make_loader(args.image_dir, 5, 5)
    make_script(model, loader)
    # onnx_export(loader, model)
