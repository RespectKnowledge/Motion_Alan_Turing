#%% ################# dataset and dataloader #########################
import os
import glob
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.transform import resize

class ACDC_Dataset(Dataset):
    def __init__(self, data_dir, rsize=(224,224,32)):
        """
        Args:
            data_dir: path to 'train' or 'val' folder containing patient files
            rsize: resize shape (H, W, D)
        """
        self.data_dir = data_dir
        self.rsize = rsize

        # collect subjects by looking for ED files
        self.subjects = sorted([
            os.path.basename(f).split("_")[0]
            for f in glob.glob(os.path.join(data_dir, "*_ED.nii.gz"))
        ])

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subj = self.subjects[index]

        # file paths
        fixed_path = os.path.join(self.data_dir, f"{subj}_ED.nii.gz")
        move_path  = os.path.join(self.data_dir, f"{subj}_ES.nii.gz")
        fixed_label_path = os.path.join(self.data_dir, f"{subj}_ED_gt.nii.gz")
        move_label_path  = os.path.join(self.data_dir, f"{subj}_ES_gt.nii.gz")

        # -------- Images --------
        fixed_img = sitk.ReadImage(fixed_path)
        move_img  = sitk.ReadImage(move_path)
        fixed_array = sitk.GetArrayFromImage(fixed_img).swapaxes(2, 0)
        move_array  = sitk.GetArrayFromImage(move_img).swapaxes(2, 0)

        fixed_resized = resize(fixed_array, self.rsize,
                               preserve_range=True, anti_aliasing=False, order=3)
        move_resized  = resize(move_array, self.rsize,
                               preserve_range=True, anti_aliasing=False, order=3)

        fixed_tensor = torch.from_numpy(fixed_resized).unsqueeze(0).float()
        move_tensor  = torch.from_numpy(move_resized).unsqueeze(0).float()

        # -------- Labels --------
        fixed_label = sitk.ReadImage(fixed_label_path)
        move_label  = sitk.ReadImage(move_label_path)
        fixed_arrayla = sitk.GetArrayFromImage(fixed_label).swapaxes(2, 0)
        move_arrayla  = sitk.GetArrayFromImage(move_label).swapaxes(2, 0)

        fixed_resized_la = resize(fixed_arrayla, self.rsize,
                                  preserve_range=True, anti_aliasing=False, order=0)
        move_resized_la  = resize(move_arrayla, self.rsize,
                                  preserve_range=True, anti_aliasing=False, order=0)

        fixed_label_tensor = torch.from_numpy(fixed_resized_la).unsqueeze(0).long()
        move_label_tensor  = torch.from_numpy(move_resized_la).unsqueeze(0).long()

        return fixed_tensor, move_tensor, fixed_label_tensor, move_label_tensor

#from torch.utils.data import DataLoader
#
#train_dir = "/mnt/ssd7tb_b/Registration_summerschool/clean_voxel_morph/dataset_split/train"
#val_dir   = "/mnt/ssd7tb_b/Registration_summerschool/clean_voxel_morph/dataset_split/val"
#
#train_dataset = ACDC_Dataset(train_dir, rsize=(224,224,32))
#val_dataset   = ACDC_Dataset(val_dir, rsize=(224,224,32))
#
#train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
#
#print("Train subjects:", len(train_dataset))
#print("Val subjects:", len(val_dataset))
#
## check one sample
#img1, img2, lab1, lab2 = train_dataset[0]
#print(img1.shape, img2.shape, lab1.shape, lab2.shape)




























