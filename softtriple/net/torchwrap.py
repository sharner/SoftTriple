from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from efficientnet_pytorch import EfficientNet
# from efficientnet_lite_pytorch import EfficientNet
# from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
# from efficientnet_lite2_pytorch_model import EfficientnetLite2ModelFile

__all__ = ['TorchWrap', 'torchwrap']


class TorchWrap(nn.Module):

    def __init__(self, backbone, dim=64):
        super(TorchWrap, self).__init__()
        self.dim = dim
        self.backbone = backbone
        self.model_ft = None
        if self.backbone == "mobilenet":
            self.model_ft = torch.hub.load(
                'pytorch/vision:v0.8.0', 'mobilenet_v2', pretrained=True)
            # SJH try this one? https://pytorch.org/tutorials/recipes/model_preparation_ios.html#get-pretrained-and-quantized-mobilenet-v2-model
            # SJH: Is there something magic about their bn-inception model or will inception_v3 work just as well?
            # model_ft = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
            num_ftrs = self.model_ft.last_channel
            self.model_ft.classifier = nn.Linear(num_ftrs, self.dim)
        elif 'efficient' in self.backbone:
            if 'lite' in self.backbone:
                weights_path = EfficientnetLite2ModelFile.get_model_file_path()
                backb = 'efficientnet-lite0'
                backb = 'efficientnet-lite2'
                self.model_ft = EfficientNet.from_pretrained(
                    backb, weights_path=weights_path)
                num_ftrs = self.model_ft._fc.in_features
                self.model_ft._fc = nn.Linear(num_ftrs, self.dim)
            # Try with efficientnet lite as well: https://github.com/lukemelas/EfficientNet-PyTorch
            if "b7" in self.backbone:
                self.model_ft = EfficientNet.from_pretrained(
                    "efficientnet-b7", self.dim)
                num_ftrs = self.model_ft._fc.in_features
                self.model_ft._fc = nn.Linear(num_ftrs, self.dim)
                self.model_ft.set_swish(memory_efficient=False)
            if "b0" in self.backbone:
                self.model_ft = EfficientNet.from_pretrained(
                    "efficientnet-b0", self.dim)
                num_ftrs = self.model_ft._fc.in_features
                self.model_ft._fc = nn.Linear(num_ftrs, self.dim)
        elif 'inceptionv3' in self.backbone:
            # model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            self.model_ft = models.inception_v3(pretrained=True)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.dim)
        # Default is ResNet50
        else:
            if "18" in self.backbone:
                print("resnet18")
                self.model_ft = models.resnet18(pretrained=True)
            elif "101" in self.backbone:
                self.model_ft = models.resnet101(pretrained=True)
            elif "152" in self.backbone:
                self.model_ft = models.resnet152(pretrained=True)
            else:
                self.model_ft = models.resnet50(pretrained=True)
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.dim)
            self.model_ft.set_swish(memory_efficient=False)
        # BNInception-style FC
        # self.embedding = nn.Linear(1024, self.dim)

    def forward(self, input):
        x = self.model_ft(input)
        if "inception" in self.backbone:
            x = x[0]  # inception_v3 returns a tuple
        # TODO: Pool like with BNInception? - need to grab the last convolution layer...
        # adaptiveAvgPoolWidth = x.shape[2]
        # x = F.avg_pool2d(x, kernel_size=int(adaptiveAvgPoolWidth))
        # x = x.view(x.size(0), -1)
        # x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def torchwrap(backbone, dim=64):
    model = TorchWrap(backbone, dim=dim)
    return model
