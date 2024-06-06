# from distutils.core import setup, find_packages
from setuptools import find_packages, setup

setup(
    name='asep',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pymongo',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'omegaconf',
        'pyyaml',
        'biopython',
        'matplotlib',
        'seaborn',
        'tqdm',
        'networkx',
        'torcheval',
        'gdown==5.0.1',
        'hydra-core==1.3.2',
        'pydantic==1.10.15',
        'pytest==8.1.1',
    ],
    # add console scripts here
    entry_points={
        'console_scripts': [
            'download-asep = asep.app.download_dataset:app',
        ]
    },
    author='chunan',
    author_email='chunan.liu@ucl.ac.uk',
)
