
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.vxm_model import VxmDenseSimple
from dataset.medical_dataset import ACDC_Dataset
from losses.losses import Grad3d, NCC, dice_val_VOI
from utils.utils import AverageMeter, adjust_learning_rate, mk_grid_img, save_checkpoint
from models.spatial_transformer import SpatialTransformer, RegisterModel

img_size = (128, 128, 16)

# -------------------- Dataset --------------------
train_dir = "/mnt/ssd7tb_b/Registration_summerschool/clean_voxel_morph/dataset_split/train"
val_dir = "/mnt/ssd7tb_b/Registration_summerschool/clean_voxel_morph/dataset_split/val"

train_loader = DataLoader(ACDC_Dataset(train_dir, rsize=img_size), batch_size=2, shuffle=True)
val_loader = DataLoader(ACDC_Dataset(val_dir, rsize=img_size), batch_size=2, shuffle=False)

# -------------------- Device & Model --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = (128, 128, 16)
model = VxmDenseSimple(inshape=img_size).to(device)

# -------------------- Loss & Optimizer --------------------
criterion_ncc = nn.MSELoss().to(device)
criterion_reg = Grad3d(penalty='l2').to(device)
weights = [1, 1.0]  # or even 5.0 depending on behavior
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)

# -------------------- Spatial Transformer --------------------

reg_model = RegisterModel(img_size, 'nearest').cuda()
reg_model_bilin = RegisterModel(img_size, 'bilinear').cuda()

# -------------------- Logging & Checkpoints --------------------
save_dir = f'vxm_2_mse_{weights[0]}_voxelmorph_{weights[1]}/'
os.makedirs('experiments/'+save_dir, exist_ok=True)
os.makedirs('logs/'+save_dir, exist_ok=True)
writer = SummaryWriter(log_dir='logs/'+save_dir)

# -------------------- GPU Info --------------------
GPU_iden = 0
GPU_num = torch.cuda.device_count()
print(f'Number of GPU: {GPU_num}')
for GPU_idx in range(GPU_num):
    print(f'     GPU #{GPU_idx}: {torch.cuda.get_device_name(GPU_idx)}')
torch.cuda.set_device(GPU_iden)
print(f'Currently using: {torch.cuda.get_device_name(GPU_iden)}')
print(f'If the GPU is available? {torch.cuda.is_available()}')

torch.manual_seed(0)

# -------------------- Training --------------------
epoch_start = 0
max_epoch = 100
best_dsc = 0
lr = 0.0001

for epoch in range(epoch_start, max_epoch):
    print(f'Training Epoch {epoch+1}/{max_epoch}')
    loss_all = AverageMeter()
    model.train()

    for idx, data in enumerate(train_loader, 1):
        # Move data to GPU
        data = [t.cuda() for t in data]
        x, y, x_seg, y_seg = data

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, max_epoch, lr)

        # ---------------- Forward pass: Fixed -> Moving ----------------
        x_in = torch.cat((x, y), dim=1)
        output, flow = model(x_in)
        def_out = reg_model([x_seg.float(), flow])

        # Loss
        loss_ncc = criterion_ncc(output, x) * weights[0]
        loss_reg = criterion_reg(flow * 10, x) * weights[1]  # scale flow for reg
        loss = loss_ncc + loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.update(loss.item(), y.numel())
        del x_in, loss

        # ---------------- Forward pass: Moving -> Fixed ----------------
        y_in = torch.cat((y, x), dim=1)
        output, flow = model(y_in)

        loss_ncc = criterion_ncc(output, y) * weights[0]
        loss_reg = criterion_reg(flow * 10, y) * weights[1]  # scale flow for reg
        loss = loss_ncc + loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.update(loss.item(), x.numel())
        del y_in

        print(f'Iter {idx}/{len(train_loader)} - Loss: {loss.item():.4f}, NCC: {loss_ncc.item():.6f}, Reg: {loss_reg.item():.6f}')

    print(f'Epoch {epoch+1} Training Loss: {loss_all.avg:.4f}')

    # -------------------- Validation --------------------
    eval_dsc = AverageMeter()
    with torch.no_grad():
        model.eval()
        for data in val_loader:
            data = [t.cuda() for t in data]
            x, y, x_seg, y_seg = data
            batch_size = x.shape[0]

            x_in = torch.cat((x, y), dim=1)

            # Forward pass
            output, flow = model(x_in)

            # Apply deformation to segmentations
            def_out = reg_model([x_seg.float(), flow])

            # Generate grid image for visualization
            grid_img = mk_grid_img(8, 1, img_size)  # single grid
            grid_img_batch = grid_img.repeat(batch_size, 1, 1, 1, 1).float().cuda()
            def_grid = reg_model_bilin([grid_img_batch, flow])

            # Dice score evaluation
            dsc = dice_val_VOI(def_out.long(), y_seg.long())
            eval_dsc.update(dsc.item(), batch_size)

    # -------------------- Save Best Model --------------------
    best_dsc = max(eval_dsc.avg, best_dsc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_dsc': best_dsc,
        'optimizer': optimizer.state_dict(),
    }, save_dir='experiments/'+save_dir, filename=f'dsc{eval_dsc.avg:.4f}.pth.tar')

    loss_all.reset()
    del def_out, def_grid, grid_img, output


writer.close()

########################################### updated training #######################################

## -------------------- Loss & Optimizer --------------------
#criterion_ncc = nn.MSELoss().to(device)
#criterion_reg = Grad3d(penalty='l2').to(device)
#
## Adjusted weights for stability
#weights = [0.8, 0.2]  # NCC vs regularization
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
#
## -------------------- Spatial Transformer --------------------
#reg_model = RegisterModel(img_size, 'nearest').cuda()
#reg_model_bilin = RegisterModel(img_size, 'bilinear').cuda()
#
## -------------------- Logging & Checkpoints --------------------
#save_dir = f'vxm_2_mse_{weights[0]}_voxelmorph_{weights[1]}/'
#os.makedirs('experiments/' + save_dir, exist_ok=True)
#os.makedirs('logs/' + save_dir, exist_ok=True)
#writer = SummaryWriter(log_dir='logs/' + save_dir)
#
## -------------------- GPU Info --------------------
#GPU_iden = 0
#GPU_num = torch.cuda.device_count()
#print(f'Number of GPU: {GPU_num}')
#for GPU_idx in range(GPU_num):
#    print(f'     GPU #{GPU_idx}: {torch.cuda.get_device_name(GPU_idx)}')
#torch.cuda.set_device(GPU_iden)
#print(f'Currently using: {torch.cuda.get_device_name(GPU_iden)}')
#print(f'If the GPU is available? {torch.cuda.is_available()}')
#
#torch.manual_seed(0)
#
## -------------------- Training --------------------
#epoch_start = 0
#max_epoch = 100
#best_dsc = 0
#lr = 1e-4
#max_flow = 0.3  # Clip flow to prevent extreme deformations
#
#for epoch in range(epoch_start, max_epoch):
#    print(f'Training Epoch {epoch + 1}/{max_epoch}')
#    loss_all = AverageMeter()
#    model.train()
#
#    # Optional: linearly decay regularization weight
#    weight_reg_epoch = weights[1] * (1 - epoch / max_epoch) + 0.05 * (epoch / max_epoch)
#
#    for idx, data in enumerate(train_loader, 1):
#        # Move data to GPU
#        data = [t.cuda() for t in data]
#        x, y, x_seg, y_seg = data
#
#        # Adjust learning rate
#        adjust_learning_rate(optimizer, epoch, max_epoch, lr)
#
#        # ---------------- Forward pass: Fixed -> Moving ----------------
#        x_in = torch.cat((x, y), dim=1)
#        output, flow = model(x_in)
#        flow = torch.clamp(flow, -max_flow, max_flow)  # Clip flow
#        def_out = reg_model([x_seg.float(), flow])
#        loss_ncc = criterion_ncc(output, x) * weights[0]
#        loss_reg = criterion_reg(flow, x) * weight_reg_epoch
#        loss = loss_ncc + loss_reg
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        loss_all.update(loss.item(), y.numel())
#        del x_in, loss
#
#        # ---------------- Forward pass: Moving -> Fixed ----------------
#        y_in = torch.cat((y, x), dim=1)
#        output, flow = model(y_in)
#        flow = torch.clamp(flow, -max_flow, max_flow)
#        loss_ncc = criterion_ncc(output, y) * weights[0]
#        loss_reg = criterion_reg(flow, y) * weight_reg_epoch
#        loss = loss_ncc + loss_reg
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        loss_all.update(loss.item(), x.numel())
#        del y_in
#
#        print(f'Iter {idx}/{len(train_loader)} - Loss: {loss.item():.4f}, NCC: {loss_ncc.item():.6f}, Reg: {loss_reg.item():.6f}')
#
#    print(f'Epoch {epoch + 1} Training Loss: {loss_all.avg:.4f}')
#
#    # -------------------- Validation --------------------
#    eval_dsc = AverageMeter()
#    with torch.no_grad():
#        model.eval()
#        for data in val_loader:
#            data = [t.cuda() for t in data]
#            x, y, x_seg, y_seg = data
#            x_in = torch.cat((x, y), dim=1)
#
#            # Generate grid image for visualization
#            grid_img = mk_grid_img(8, 1, img_size)
#
#            # Forward pass
#            output, flow = model(x_in)
#            flow = torch.clamp(flow, -max_flow, max_flow)
#            def_out = reg_model([x_seg.float(), flow])
#            def_grid = reg_model_bilin([grid_img.float(), flow])
#            dsc = dice_val_VOI(def_out.long(), y_seg.long())
#            eval_dsc.update(dsc.item(), x.size(0))
#
#    # -------------------- Save Best Model --------------------
#    best_dsc = max(eval_dsc.avg, best_dsc)
#    save_checkpoint({
#        'epoch': epoch + 1,
#        'state_dict': model.state_dict(),
#        'best_dsc': best_dsc,
#        'optimizer': optimizer.state_dict(),
#    }, save_dir='experiments/' + save_dir, filename=f'dsc{eval_dsc.avg:.4f}.pth.tar')
#
#    loss_all.reset()
#    del def_out, def_grid, grid_img, output, flow
#
#writer.close()

