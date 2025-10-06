# VoxelMorph (PyTorch) Environment Setup Guide

This document describes the steps to create a **conda-based Python environment** 
and install all required dependencies for running **VoxelMorph** using **PyTorch**.

---

# Create and Activate Conda Environment

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


