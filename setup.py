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
        'torch==1.7.0',
        'torchvision==0.8.0',
        'natsort==7.1.0',
        'pyarrow==2.0.0',
        'efficientnet_pytorch==0.7.1',
        'efficientnet-lite-pytorch==0.1.0',
        'efficientnet-lite0-pytorch-model==0.1.0'
    ],
    entry_points = {'console_scripts': ['softtriple=softtriple.train:main']}
)
