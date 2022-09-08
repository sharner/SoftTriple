from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softtriple',
    version='0.0.0',
    description='SoftTriple network',
    author = 'Soren Harner',
    packages=['softtriple'],
    ext_modules=[],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch==1.11.0',
        'torchvision==0.12.0',
        'natsort==7.1.0',
        'pyarrow==9.0.0',
        'scikit-learn==1.1.2',
        'efficientnet_pytorch==0.7.1',
        'efficientnet-lite-pytorch==0.1.0',
        'efficientnet-lite0-pytorch-model==0.1.0',
        'efficientnet-lite2-pytorch-model==0.1.0'
    ],
    entry_points = {'console_scripts': ['softtriple=softtriple.lj_train:main']},
    include_package_data=True
)
