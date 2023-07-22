"""
    PyTorch Package for SoftTriple Loss

    Reference
    ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

    Copyright@Alibaba Group

"""

import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, datasets, models

import torch.nn as nn
from PIL import Image
from softtriple import loss
from softtriple.evaluation import evaluation
from softtriple import net

import timm.data.auto_augment
from timm.data.transforms import RandomResizedCropAndInterpolation

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--dim', default=64, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('--la', default=20, type=float,
                    help='lambda')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--tau', default=0.2, type=float,
                    help='tau')
parser.add_argument('--margin', default=0.01, type=float,
                    help='margin')
parser.add_argument('-C', default=98, type=int,
                    help='C')
parser.add_argument('-K', default=10, type=int,
                    help='centers')
parser.add_argument('--backbone', default='BN-Inception', type=str,
                    help='type of model to use: "resnet" for Resnet152, "mobilenet" for Mobilenet_v2, "efficientb7" + "efficientb0" for Efficient Net B0 and B7, "efficientlite" for Efficient Net Lite')
parser.add_argument('--rand_config', default='rand-mstd1',
                    help='Random augment configuration string')
parser.add_argument('--max_level', default=None, type=float,
                    help='change rand augmentation max')

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))


def make_backbone(dims, backbone, pretrained=True):
    if "BN-Inception" in backbone:
        # for BN-Inception, default
        model_ft = net.bninception(dims)
    else:
        model_ft = net.torchwrap(backbone, dims, pretrained)
    return model_ft


def main():
    args = parser.parse_args()

    if args.max_level:
        timm.data.auto_augment._LEVEL_DENOM = args.max_level

    # create model
    print("Training model with backbone", args.backbone)
    model = make_backbone(args.dim, args.backbone)

    # SJH: uncomment to run on a selected GPU
    #
    b = True
    if args.backbone == "mobilenet":  # mobilenet
        torch.cuda.set_device(args.gpu)
        b = False
    model = model.cuda()
    # SJH: uncomment to run on multiple GPUs - works for ResNet not MobileNet
    if b is True:  # false for mobilenet
        model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = loss.SoftTriple(
        args.la, args.gamma, args.tau, args.margin, args.dim, args.C, args.K).cuda()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.modellr},
                                  {"params": criterion.parameters(), "lr": args.centerlr}],
                                 eps=args.eps, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # load data
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')

    # Use this with BN-Inception
    if args.backbone == "BN-Inception":
        normalize = transforms.Normalize(mean=[104., 117., 128.],
                                         std=[1., 1., 1.])
    # SJH for ResNet and EfficientNet
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if args.backbone == "BN-Inception":
        input_dim_resize = 256
        input_dim_crop = 224
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Lambda(RGB2BGR),  # SJH BN-Inception is BGR
                # SJH was 224 for bn-inception and mobilenet and 299 for pytorch inception
                transforms.RandomResizedCrop(input_dim_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # SJH BN-Inception needs scale
                transforms.Lambda(lambda x: x.mul(255)),
                normalize,
            ]))
    else:
        input_dim_resize = 1024
        input_dim_crop = 598
        # input_dim_resize = 512
        # input_dim_crop = 299

        # TODO: investigate this?
        # For EfficientNet with advprop
        # normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
        if args.rand_config:
            print("Using random augmentation...")
            # note mean is 255 * (0.485, 0.456, 0.406).  TODO define
            # mean in one spot to make sure normalize and rand augment
            # have same mean.
            rand_tfm = timm.data.auto_augment.rand_augment_transform\
                (config_str=args.rand_config,
                 hparams={'img_mean': (124, 116, 104)})
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    RandomResizedCropAndInterpolation(input_dim_crop),
                    transforms.RandomHorizontalFlip(),
                    rand_tfm,
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            print("Not using random augmentation...")
            # note mean is 255 * (0.485, 0.456, 0.406).  TODO define
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(input_dim_crop),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.backbone == "BN-Inception":
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Lambda(RGB2BGR),
                transforms.Resize(input_dim_resize),
                transforms.CenterCrop(input_dim_crop),
                transforms.ToTensor(),
                # SJH BN-Inception needs scale
                transforms.Lambda(lambda x: x.mul(255)),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        print(testdir)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(input_dim_resize),
                transforms.CenterCrop(input_dim_crop),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    best_nmi = 0
    for epoch in range(args.start_epoch, args.epochs):
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, args)

        # Run validation
        nmi, recall = validate(test_loader, model, args)
        print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
              .format(recall=recall, nmi=nmi))

        # Save the best model
        if nmi > best_nmi:
            best_nmi = nmi
            print("Saving new best model!")
            fn = "{}.pth".format(f"best_model_{epoch}")
            torch.save(model.state_dict(), fn)
            print("Model saved to", fn)

    # evaluate on validation set
    nmi, recall = validate(test_loader, model, args)
    print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
          .format(recall=recall, nmi=nmi))

    # Save the model
    print("Saving model!")
    fn = "{}.pth".format("last_model")
    torch.save(model.state_dict(), fn)
    print("Model saved to", fn)


num_avg_iter = 500


def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode

    model.train()
    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    run_loss = 0
    for i, (input, target) in enumerate(train_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        run_loss += loss.item()
        if i % num_avg_iter == 0:
            print('Training loss running avg', run_loss/float(num_avg_iter))
            run_loss = 0

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(test_loader, model, args):
    # switch to evaluation mode
    model.eval()
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, target))
    nmi, recall = evaluation(
        testdata.numpy(), testlabel.numpy(), [1, 2, 4, 8])
    return nmi, recall


def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch+1) % 20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate


if __name__ == '__main__':
    main()
