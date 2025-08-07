import torch
import torch.utils.data as data
import glob
import os
from common import *
import numpy as np
import nibabel as nib
import os
import pywt
from torchio import transforms as T
def wavelet(image):
    image_wave = pywt.dwtn(image, 'haar')

    LLL = image_wave['aaa']
    LLH = image_wave['aad']
    LHL = image_wave['ada']
    LHH = image_wave['add']
    HLL = image_wave['daa']
    HLH = image_wave['dad']
    HHL = image_wave['dda']
    HHH = image_wave['ddd']

    LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255
    LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
    LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
    LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
    HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
    HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
    HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
    HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

    merge1 = LLH + LHL + LHH + HLL + HLH + HHL + HHH
    merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255
    return LLL, merge1






#BCHW order
class MDBDataset(data.Dataset):

    def __init__(self, root_path, crop_size=crop_size, mode='train', fold=1):
        all_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
                    '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
        if mode == 'train':
            if fold == 1:
                all_list = all_list[0:20]
            if fold == 2:
                all_list = all_list[0:15] + all_list[20:26]
            if fold == 3:
                all_list = all_list[0:10] + all_list[15:26]
            if fold == 4:
                all_list = all_list[0:5] + all_list[10:26]
            if fold == 5:
                all_list = all_list[5:26]
        self.pa27 = []
        self.lab = []
        for i in all_list:
            self.path27 = os.path.join(root_path, 'train' + i, 'tse_native_chunk_left.nii.gz')
            self.path_label = os.path.join(root_path, 'train' + i, 'tse_native_chunk_left_seg_6.nii.gz')
            self.pa27.append(self.path27)
            self.lab.append(self.path_label)
            self.crop_size = crop_size
            self.crop_size_small = crop_size_small
            self.mode = mode


    def __getitem__(self, index):
        image20 = nib.load(self.pa27[index]).get_fdata()
        image2 = np.expand_dims(image20, axis=0)
        self.image = np.expand_dims(image2, axis=0)
        self.label0 = nib.load(self.lab[index]).get_fdata()
        self.label0 = np.asarray(self.label0)
        self.label = np.expand_dims(self.label0, axis=0)
        self.label = np.expand_dims(self.label, axis=0)

        _, _, C, H, W = self.image.shape
        if (self.mode=='train'):
            cx = random.randint(0, C - self.crop_size[0]-1)
            cy = random.randint(0, H - self.crop_size[1]-1)
            cz = random.randint(0, W - self.crop_size[2]-1)

        self.data7 = self.image[:, :, cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]
        wave_T2_L1, wave_T2_H1 = wavelet(np.squeeze(self.data7[:, :, :, :, :]))
        wave_T2_L2, wave_T2_H2 = wavelet(wave_T2_L1)
        wave_T2_L3, wave_T2_H3 = wavelet(wave_T2_L2)
        wave_T2_L1 = np.expand_dims(wave_T2_L1, axis=0)
        wave_T2_H1 = np.expand_dims(wave_T2_H1, axis=0)
        wave_T2_L2 = np.expand_dims(wave_T2_L2, axis=0)
        wave_T2_H2 = np.expand_dims(wave_T2_H2, axis=0)
        wave_T2_L3 = np.expand_dims(wave_T2_L3, axis=0)
        wave_T2_H3 = np.expand_dims(wave_T2_H3, axis=0)

        self.wave1 = np.concatenate((wave_T2_L1, wave_T2_H1), axis=0)
        self.wave2 = np.concatenate((wave_T2_L2, wave_T2_H2), axis=0)
        self.wave3 = np.concatenate((wave_T2_L3, wave_T2_H3), axis=0)


        self.wave1 = np.expand_dims(self.wave1, axis=0)
        self.wave2 = np.expand_dims(self.wave2, axis=0)
        self.wave3 = np.expand_dims(self.wave3, axis=0)
        self.data_crop_wave1 = self.wave1
        self.data_crop_wave2 = self.wave2
        self.data_crop_wave3 = self.wave3
        self.label_crop = self.label[:, :,  cx: cx + self.crop_size[0], cy: cy + self.crop_size[1], cz: cz + self.crop_size[2]]


        # ------End random crop-------------
        return (torch.from_numpy(self.data7[0, :, :, :, :]).float(),
                torch.from_numpy(self.data_crop_wave1[0, :, :, :, :]).float(),
                torch.from_numpy(self.data_crop_wave2[0, :, :, :, :]).float(),
                torch.from_numpy(self.data_crop_wave3[0, :, :, :, :]).float(),
                torch.from_numpy(self.label_crop[0, 0, :, :, :]).long())

    def __len__(self):
        return len(self.lab)
