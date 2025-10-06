# VoxelMorph (PyTorch) Environment Setup Guide and training

This document describes the steps to create a **conda-based Python environment** 
and install all required dependencies for running **VoxelMorph** using **PyTorch**.


## Create and Activate Conda Environment and run training

```bash
conda create -n voxelmorph_rig python=3.10 -y
conda activate voxelmorph_rig

## Install PyTorch

## Choose the CUDA version that matches your system (below uses CUDA 12.1).
If you are using CPU only, you can omit the --index-url flag.

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

## Install Core Dependencies
pip install nibabel h5py scikit-image matplotlib tqdm
pip install tensorboard
pip install SimpleITK
pip install natsort

#Please set dataset, model save paths etc

/dataset_split/
├── train/
│   ├── patient002_ED.nii.gz
│   ├── patient002_ES.nii.gz
│   ├── patient002_ED_gt.nii.gz
│   ├── patient002_ES_gt.nii.gz
│   ├── ...
│
└── val/
    ├── patient072_ED.nii.gz
    ├── patient072_ES.nii.gz
    ├── patient072_ED_gt.nii.gz
    ├── patient072_ES_gt.nii.gz
    └── ...
########## run
python3 train.py
