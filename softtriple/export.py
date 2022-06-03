import argparse, sys, os, io, time, copy
import torch
import onnx
import torch.nn as nn
from typing import List, Dict, Tuple
from PIL import Image
from torchvision import transforms
from torch.utils import data
from torchvision import models

# import pycuda.driver as cuda
# import pycuda.autoinit
import tensorrt as trt
from efficientnet_pytorch import EfficientNet

def load_model(model_checkpoint: str) -> nn.Module:
    model = torch.load(model_checkpoint)
    model.cuda(0)
    model.eval()
    return model

def load_resnet():
    return models.resnet50(pretrained=True)

def save_torch(model, base_name):
    print("Saving model!")
    fn = "{}.pth".format(base_name)
    torch.save(model, fn)
    print("Model saved to", fn)

def onnx_export(model: nn.Module, fn: str):
    # switch to evaluation mode
    labels = []
    # dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 598, 598)
    # dummy_input = torch.randn(1, 3, 299, 299)
    dummy_input = dummy_input.cuda(0)
    print(dummy_input.shape)
    # SJH: Needed to export EffNetB7 https://github.com/lukemelas/EfficientNet-PyTorch/issues/91
    model.module.model_ft.set_swish(memory_efficient=False)
    # SJH: use model when not trained with DataParallel iterator
    # SJH: the EffNetB7 and BNInception needs opcode 10
    # SJH: BN-Inception requires opcode 9 to convert successfully to TFLite
    opset=10
    print('Exporting with opset', opset)
    torch.onnx.export(model.module, dummy_input, fn,
                      verbose=True,
                      input_names=['input'], output_names=['output'],
                      opset_version=opset,
                      export_params=True)
    print('Exported model', fn)

def test_onnx_model(fn: str):
    onnx_model = onnx.load(fn)
    onnx.checker.check_model(onnx_model)

TRT_LOGGER = trt.Logger()

# TODO: get this to work
def trt_export(onnx_fn: str, trt_fn: str):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_fn, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        # Allow TRT to use up tu 1GB memory
        builder.max_workspace_size = 1 << 30
        # allow one image per batch (TODO: increase this?)
        builder.max_batch_size = 1
        # use FP16 mode
        builder.fp16_mode = True
        # mark the output layer (https://github.com/NVIDIA/TensorRT/issues/183)
        # print("Number of layers", network.num_layers)
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        print('Building an engine...')
        engine = builder.build_cuda_engine(network)
        # context = engine.create_execution_context()
        print("Completed creating Engine")
        # engine.save(trt_fn)
        print("Engine saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a SoftTriple model')
    parser.add_argument('--model', type=str,  help='Pickle file with saved model')
    args = parser.parse_args()
    print('Model', args.model)
    model = load_model(args.model)
    # model = load_resnet()
    ONNX_FILE_NAME="model_n1.onnx"
    TRT_FILE_NAME="model_n1.plan"
    onnx_export(model, ONNX_FILE_NAME)
    test_onnx_model(ONNX_FILE_NAME)
    # trt_export(ONNX_FILE_NAME, TRT_FILE_NAME)
