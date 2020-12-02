
# SoftTriple Loss

PyTorch Implementation for Our ICCV'19 Paper: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

## Usage: Train on Cars196
Here is an example of using this package.

1. Obtain dataset
```
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz
tar -xf car_ims.tgz
```

2. Generate train/test sets
```
python genCars.py
```

3. Learn 64-dimensional embeddings
```
python train.py --gpu 0 --dim 64 -C 98 --freeze_BN [folder with train and test folders]
```

## Requirements
* Python 3.7
* PyTorch 1.1
* scikit-learn 0.20.1

    
## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2019striple,
  author    = {Qi Qian and
               Lei Shang and
               Baigui Sun and
               Juhua Hu and
               Hao Li and
               Rong Jin},
  title     = {SoftTriple Loss: Deep Metric Learning Without Triplet Sampling},
  booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2019},
  year      = {2019}
}
```

## Building the Container (SJH)

```{sh}
docker build -t soft-triple-n1:latest .
```

## Set up for training

Get the training file from GS_RESULTS.

```{sh}
unzip n1model.zip && mv pklfiles train
```

Place some of the classes in test:

```{sh}
mkdir test
mv train/<choose some directories> test
```

## Train

Start the training job. Start it on GPU 1 so that GPU 0 is available for other things.

```{sh}
NUM_CLASSES=$(ls -1 train | wc -l)
python train.py --batch-size 128 --gpu 1 --dim 512 -C $NUM_CLASSES --freeze_BN .
```

