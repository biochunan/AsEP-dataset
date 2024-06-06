#!/bin/zsh

# init conda
conda init zsh

# source .zshrc to activate conda
source $HOME/.zshrc

# turn off conda auto-activate base
conda config --set auto_activate_base false

conda create -n walle python=3.10 -y && conda activate walle  # 3.10 because package openmm-7.7.0-py310h0df5d99_1 requires python >=3.10,<3.11.0a0

# install pytorch 2.1.1
conda install -c pytorch -c nvidia -y pytorch==2.1.1 torchvision torchaudio pytorch-cuda=12.1
# install torch_geometric
pip install torch_geometric
# Optional dependencies:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.1+cu121.html
pip install torcheval==0.0.6

# install esm
pip install fair-esm

# install igfold from git repo
git clone git@github.com:Graylab/IgFold.git
pip install IgFold
rm -rf IgFold
conda install -y -c conda-forge openmm==7.7.0 pdbfixer  # cudatoolkit-11.7.0 is required by openmm
conda install -y -c bioconda abnumber

# extra pakcages
# sudo apt insatll -y dssp # installs the latest version of dssp, executable at /usr/bin/dssp, though graphein is not fully supported
conda install -c salilab dssp -y # requires libboost 1.73.0 explicitly, installs mkdssp version 3.0.0, executable at /home/vscode/.conda/envs/walle/bin/mkdssp
conda install -c anaconda libboost==1.73.0 -y  # required by dssp
pip install loguru biopandas omegaconf pyyaml tqdm wandb \
    torcheval torchmetrics gdown 'graphein[extras]' docker \
    seaborn matplotlib
cd /workspaces/Antibody-specific-epitope-prediction-2.0 && pip install .
conda install -y -c bioconda clustalo  # clustal omega for pairwise alignment

# install seqres2atmseq
pip install git+https://github.com/biochunan/seqres2atmseq.git

# cleanup
conda clean -a -y && \
pip cache purge && \
sudo apt autoremove -y
