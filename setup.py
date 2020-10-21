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
        'torch>=1.0.0a0',
        'torchvision'    
    ],
    entry_points = {'console_scripts': ['softtriple=softtriple.train:main']}
)
