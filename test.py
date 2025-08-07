import nibabel as nib
from MedicalUtil import *
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import pywt
import torch
import monai.transforms as monai_transforms
flip_transform = monai_transforms.RandFlip(prob=1.0, spatial_axis=0)
xstep = 8
ystep = 8
zstep = 8
from common import *

all_mean = []
all_hd95 = []
DSC_total = []
hd95_total = []
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = MWFNet(1, 7).to(device)
checkpoints = '/home/student9/only_segmentation_project/only_7T/only7T2/val_data2017/fold5/_00122_0.792965_model.pth'
print('Checkpoint: ', checkpoints)

saved_state_dict = torch.load(checkpoints)
net.load_state_dict(saved_state_dict)
net.eval()
val_list_fold1 = ['020', '021', '022', '023', '024', '025']
val_list_fold2 = ['015', '016', '017', '018', '019']
val_list_fold3 = ['010', '011', '012', '013', '014']
val_list_fold4 = ['005', '006', '007', '008', '009']
val_list_fold5 = ['000', '001', '002', '003', '004']

val_list_fold_all = ['21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32',
                    '33', '34', '35', '36', '37', '38', '39', '40']
path = r'/home/student9/SC-GAN/ashs_atlas_umcutrecht_7t_20170810/train'
save_paths_time_metric = r'/home/student9/only_segmentation_project/test_image/time_metric'
save_paths = r'/home/student9/only_segmentation_project/test_image/only7T2_val_data2017_vnet'
for ss in val_list_fold5:
    val_path1 = os.path.join(path, 'train' + ss, 'tse_native_chunk_left.nii.gz')
    lab_path = os.path.join(path, 'train' + ss, 'tse_native_chunk_left_seg_6.nii.gz')
    image10 = nib.load(val_path1).get_fdata()
    image1 = np.expand_dims(image10, axis=0)
    image1 = np.expand_dims(image1, axis=0)
    label = nib.load(lab_path).get_fdata()
    seg_save = np.asarray(label)
    label_affine = nib.load(lab_path).affine
    with torch.no_grad():
        image1 = torch.from_numpy(image1).float().to(device)
        _, _, C, H, W = image1.shape
        deep_slices = np.arange(0, C - crop_size[0] + xstep, xstep)
        height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
        width_slices = np.arange(0, W - crop_size[2] + zstep, zstep)
        whole_pred = np.zeros((1,) + (7,) + image1.shape[2:])
        count_used = np.zeros((image1.shape[2], image1.shape[3], image1.shape[4])) + 1e-5
        for i in range(len(deep_slices) - 1):
            for j in range(len(height_slices) - 1):
                for k in range(len(width_slices) - 1):
                    deep = deep_slices[i]
                    height = height_slices[j]
                    width = width_slices[k]

                    image_crop = image1[:, :, deep: deep + crop_size[0],
                                 height: height + crop_size[1],
                                 width: width + crop_size[2]]

                    image_crop1 = image_crop.data.cpu().numpy()
                    wave_T2_L1, wave_T2_H1 = wavelet(np.squeeze(image_crop1[:, :, :, :, :]))
                    wave_T2_L2, wave_T2_H2 = wavelet(wave_T2_L1)
                    wave_T2_L3, wave_T2_H3 = wavelet(wave_T2_L2)
                    wave_T2_L1 = np.expand_dims(wave_T2_L1, axis=0)
                    wave_T2_H1 = np.expand_dims(wave_T2_H1, axis=0)
                    wave_T2_L2 = np.expand_dims(wave_T2_L2, axis=0)
                    wave_T2_H2 = np.expand_dims(wave_T2_H2, axis=0)
                    wave_T2_L3 = np.expand_dims(wave_T2_L3, axis=0)
                    wave_T2_H3 = np.expand_dims(wave_T2_H3, axis=0)

                    wave1 = np.concatenate((wave_T2_L1, wave_T2_H1), axis=0)
                    wave2 = np.concatenate((wave_T2_L2, wave_T2_H2), axis=0)
                    wave3 = np.concatenate((wave_T2_L3, wave_T2_H3), axis=0)

                    wave1 = np.expand_dims(wave1, axis=0)
                    wave2 = np.expand_dims(wave2, axis=0)
                    wave3 = np.expand_dims(wave3, axis=0)

                    wave1 = torch.from_numpy(wave1).float().to(device)
                    wave2 = torch.from_numpy(wave2).float().to(device)
                    wave3 = torch.from_numpy(wave3).float().to(device)
                    outputs, _, _, _ = net(image_crop, wave1, wave2, wave3)
                    whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                    height: height + crop_size[1],
                    width: width + crop_size[2]] += outputs.data.cpu().numpy()

                    count_used[deep: deep + crop_size[0],
                    height: height + crop_size[1],
                    width: width + crop_size[2]] += 1

    whole_pred = whole_pred / count_used
    predicted_val = np.argmax(whole_pred, axis=1)
    whole_pred = predicted_val[0, :, :, :]

    targets_val = label
    dsc1 = []
    hd_95 = []
    for j in range(1, 7):  # ignore Background 0
        dsc_i = dice(whole_pred, targets_val, j)
        dsc1.append(dsc_i)
        whole_pred1 = whole_pred == j
        targets_val1 = targets_val == j
        hd_95_i = hd95(whole_pred1, targets_val1, 0.4)
        hd_95.append(hd_95_i)
    dsc1_mean = np.mean(dsc1)
    all_mean.append(dsc1_mean)

    hd_95_mean = np.mean(hd_95)
    all_hd95.append(hd_95_mean)

    print(ss, dsc1, 'mean: ', dsc1_mean)
    print(ss, hd_95, 'mean: ', hd_95_mean)
    DSC_total.append(dsc1)
    hd95_total.append(hd_95)

    save_path = os.path.join(save_paths, '%s_test_left.nii.gz' % ss)
    seg_save[:, :, :] = whole_pred[:, :, :]
    seg = nib.Nifti1Image(seg_save, label_affine)
    nib.save(seg, save_path)
print('---------', np.mean(all_mean))
print('---------', np.mean(all_hd95))
#
df = pd.DataFrame(DSC_total)
df.to_excel(os.path.join(save_paths_time_metric, 'metric_dsc.xlsx'), index=False)
df1 = pd.DataFrame(hd95_total)
df1.to_excel(os.path.join(save_paths_time_metric, 'metric_hd.xlsx'), index=False)



