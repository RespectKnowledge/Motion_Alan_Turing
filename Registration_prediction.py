# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 08:27:52 2025

@author: aq22
"""
#%% Prediction file
import os
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from models.vxm_model import VxmDenseSimple
from models.spatial_transformer import RegisterModel
from utils.utils import dice_val_substruct, dice_val

# -------------------- Utility Functions --------------------
import pystrum.pynd.ndutils as nd
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
    
# -------------------- Model Setup --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = (128,128,16)
model = VxmDenseSimple(inshape=img_size).to(device)
reg_model = RegisterModel(img_size, 'nearest').to(device)

# -------------------- Load checkpoint --------------------
checkpoint_path = '/content/drive/MyDrive/Summer_School_AT2025/trained_weights/dsc0.6097.pth.tar'  # adjust your checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded trained model.")
else:
    raise FileNotFoundError(f"{checkpoint_path} not found!")

model.eval()

val_dir = "/content/drive/MyDrive/Summer_School_AT2025/val"

################ create output paths ######################
output_dir = "/content/drive/MyDrive/Summer_School_AT2025/output_model_voxmorph/"
save_pathm = os.path.join(output_dir, 'moved')
save_pathf = os.path.join(output_dir, 'flowf')
save_pathjf = os.path.join(output_dir, 'jfield')
save_pathseg = os.path.join(output_dir, 'segout')
fixedimpath = os.path.join(output_dir, 'fixed')
movedimpath = os.path.join(output_dir, 'moving')
fixedlabelpa = os.path.join(output_dir, 'fixed_label')
movedlabelpa = os.path.join(output_dir, 'move_label')

for p in [save_pathm, save_pathf, save_pathjf, save_pathseg,
          fixedimpath, movedimpath, fixedlabelpa, movedlabelpa]:
    os.makedirs(p, exist_ok=True)
    
# -------------------- Process Patients --------------------
import os
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
# -------------------- Process Patients --------------------
patients = sorted(list(set([f.split('_')[0] for f in os.listdir(val_dir)])))

for pat in patients:
    # Paths
    fixed_path = os.path.join(val_dir, f'{pat}_ED.nii.gz')
    moving_path = os.path.join(val_dir, f'{pat}_ES.nii.gz')
    fixed_label_path = os.path.join(val_dir, f'{pat}_ED_gt.nii.gz')
    moving_label_path = os.path.join(val_dir, f'{pat}_ES_gt.nii.gz')

    # Read images with SimpleITK for array processing
    fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path)).swapaxes(0, 2)
    moving_img = sitk.GetArrayFromImage(sitk.ReadImage(moving_path)).swapaxes(0, 2)
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_path)).swapaxes(0, 2)
    moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_path)).swapaxes(0, 2)

    # Read images with nibabel to get original affine
    fixed_img_nii = nib.load(fixed_path)
    moving_img_nii = nib.load(moving_path)
    fixed_label_nii = nib.load(fixed_label_path)
    moving_label_nii = nib.load(moving_label_path)

    # Resize to target img_size
    fixed_img_r = resize(fixed_img, img_size, preserve_range=True, anti_aliasing=False, order=3)
    moving_img_r = resize(moving_img, img_size, preserve_range=True, anti_aliasing=False, order=3)
    fixed_label_r = resize(fixed_label, img_size, preserve_range=True, anti_aliasing=False, order=0)
    moving_label_r = resize(moving_label, img_size, preserve_range=True, anti_aliasing=False, order=0)

    # Save resized images and labels
    nib.save(nib.Nifti1Image(fixed_img_r.astype(np.float32), fixed_img_nii.affine),
             os.path.join(fixedimpath, f'{pat}_fixed.nii.gz'))
    nib.save(nib.Nifti1Image(moving_img_r.astype(np.float32), moving_img_nii.affine),
             os.path.join(movedimpath, f'{pat}_moving.nii.gz'))
    nib.save(nib.Nifti1Image(fixed_label_r.astype(np.uint8), fixed_label_nii.affine),
             os.path.join(fixedlabelpa, f'{pat}_fixed_label.nii.gz'))
    nib.save(nib.Nifti1Image(moving_label_r.astype(np.uint8), moving_label_nii.affine),
             os.path.join(movedlabelpa, f'{pat}_moving_label.nii.gz'))

    # Convert to torch tensors
    x = torch.from_numpy(fixed_img_r).unsqueeze(0).unsqueeze(0).float().to(device)
    y = torch.from_numpy(moving_img_r).unsqueeze(0).unsqueeze(0).float().to(device)
    x_seg = torch.from_numpy(fixed_label_r).unsqueeze(0).long().to(device)
    y_seg = torch.from_numpy(moving_label_r).unsqueeze(0).long().to(device)

    # Forward pass
    with torch.no_grad():
        x_in = torch.cat((x, y), dim=1)
        x_def, flow = model(x_in)

        # Save flow
        flow_np = flow.squeeze(0).cpu().numpy().astype(np.float32)
        np.save(os.path.join(save_pathf, f'{pat}_flow.npy'), flow_np)
        nib.save(nib.Nifti1Image(flow_np.transpose(3, 0, 1, 2), fixed_img_nii.affine),
                 os.path.join(save_pathf, f'{pat}_flow.nii.gz'))

        # Jacobian
        jac_det = jacobian_determinant_vxm(flow_np).astype(np.float32)
        np.save(os.path.join(save_pathjf, f'{pat}_jacobian.npy'), jac_det)
        nib.save(nib.Nifti1Image(jac_det, fixed_img_nii.affine),
                 os.path.join(save_pathjf, f'{pat}_jacobian.nii.gz'))

        # Deformed segmentation
        y_seg_oh = nn.functional.one_hot(y_seg, num_classes=4).permute(0, 4, 1, 2, 3).float()
        x_segs = []
        for i in range(4):
            def_seg = reg_model([y_seg_oh[:, i:i+1], flow])
            x_segs.append(def_seg)
        x_segs = torch.cat(x_segs, dim=1)
        def_out = torch.argmax(x_segs, dim=1)
        def_out_np = def_out.squeeze(0).cpu().numpy().astype(np.uint8)
        np.save(os.path.join(save_pathseg, f'{pat}_def_seg.npy'), def_out_np)
        nib.save(nib.Nifti1Image(def_out_np, fixed_img_nii.affine),
                 os.path.join(save_pathseg, f'{pat}_def_seg.nii.gz'))

        # Deformed image
        moved_np = x_def.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        np.save(os.path.join(save_pathm, f'{pat}_moved.npy'), moved_np)
        nib.save(nib.Nifti1Image(moved_np, fixed_img_nii.affine),
                 os.path.join(save_pathm, f'{pat}_moved.nii.gz'))

    print(f"Processed {pat}")
