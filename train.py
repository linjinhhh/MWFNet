import torch.utils.data as dataloader
from dataloder_only7t2 import MDBDataset
import torch.optim as optim
from common import *
import torch
import torch.nn as nn
import nibabel as nib
from dataloder_only7t2 import wavelet
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xstep = 16
ystep = 16 # 16
zstep = 16 # 16
Diceloss = DiceLoss(aux=False)
dsc_sav = 0.7

model_S = MWFNet(1, 7).to(device)
criterion_S = nn.CrossEntropyLoss().cuda()

optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.6)

fold = 5
# --------------Start Training and Validation ---------------------------
if __name__ == '__main__':
    #-----------------------Training--------------------------------------
    mri_data_train = MDBDataset("/home/student9/SC-GAN/ashs_atlas_umcutrecht_7t_20170810/train", mode='train', fold=fold)
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=batch_train, shuffle=True)
    print('Rate | epoch  | Loss seg | subject |  DSC_val')
    for epoch in range(1, num_epoch+1):
        scheduler_S.step(epoch)
        model_S.train()
        for i, data in enumerate(trainloader):
            image7, images_wave_1, images_wave_2, images_wave_3, targets = data
            images_wave_1 = images_wave_1.to(device)
            images_wave_2 = images_wave_2.to(device)
            images_wave_3 = images_wave_3.to(device)
            image = image7.to(device)
            targets = targets.to(device)
            optimizer_S.zero_grad()
            outputs1, outputs2, outputs3, outputs4 = model_S(image, images_wave_1, images_wave_2, images_wave_3)
            loss_seg = Diceloss(outputs1, targets) + (Diceloss(outputs2, targets) + Diceloss(outputs3, targets) + Diceloss(outputs4, targets))*0.5
            # -----------------------loss---------------------------------
            loss_seg.backward()
            optimizer_S.step()

        # -----------------------Validation------------------------------------
        # no update parameter gradients during validation
        with torch.no_grad():
            all_list = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
                        '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
            val_list_fold1 = all_list[0:20]
            val_list_fold2 = all_list[0:15] + all_list[20:26]
            val_list_fold3 = all_list[0:10] + all_list[15:26]
            val_list_fold4 = all_list[0:5] + all_list[10:26]
            val_list_fold5 = all_list[5:26]
            # val_list_fold1 = ['020', '021', '022', '023', '024', '025']
            # val_list_fold2 = ['015', '016', '017', '018', '019']
            # val_list_fold3 = ['010', '011', '012', '013', '014']
            # val_list_fold4 = ['005', '006', '007', '008', '009']
            # val_list_fold5 = ['000', '001', '002', '003', '004']
            # val_list_fold6 = ['17', '18', '19', '20']
            path = r'/home/student9/SC-GAN/ashs_atlas_umcutrecht_7t_20170810/train'
            dsc_mean = []
            if fold == 1:
                val_list = random.sample(val_list_fold1, 1)
            elif fold == 2:
                val_list = random.sample(val_list_fold2, 1)
            elif fold == 3:
                val_list = random.sample(val_list_fold3, 1)
            elif fold == 4:
                val_list = random.sample(val_list_fold4, 1)
            elif fold == 5:
                val_list = random.sample(val_list_fold5, 1)
            for v in val_list:
                val_path12 = os.path.join(path, 'train' + v, 'tse_native_chunk_left.nii.gz')
                image102 = nib.load(val_path12).get_fdata()
                image2 = np.expand_dims(image102, axis=0)
                image1 = np.expand_dims(image2, axis=0)
                lab_path = os.path.join(path, 'train' + v, 'tse_native_chunk_left_seg_6.nii.gz')
                img_label = nib.load(lab_path).get_fdata()
                aff = nib.load(lab_path).affine
                image1 = torch.from_numpy(image1).float().to(device)
                targets_val = img_label
                _, _, C, H, W = image1.shape
                targets_val = torch.from_numpy(targets_val[:, :, :]).long()
                model_S.eval()
                targets_val = targets_val.to(device)
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
                            outputs, _, _, _ = model_S(image_crop, wave1, wave2, wave3)
                            whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2]] += outputs.data.cpu().numpy()

                            count_used[deep: deep + crop_size[0],
                            height: height + crop_size[1],
                            width: width + crop_size[2]] += 1

                whole_pred = whole_pred / count_used
                predicted_val = np.argmax(whole_pred, axis=1)
                targets_val = targets_val.data.cpu().numpy()
                predicted_val = predicted_val[0, :, :, :]
                # print(predicted_val.shape, targets_val.shape)
                dsc = []
                for j in range(1, 7):  # ignore Background 0
                    dsc_i = dice(predicted_val, targets_val, j)
                    dsc.append(dsc_i)
                dsc_all = np.mean(dsc)
                dsc_mean.append(dsc_all)
                # -------------------Debug-------------------------
                for param_group in optimizer_S.param_groups:
                    for param_group in optimizer_S.param_groups:
                        print('%0.6f | %6d |%0.5f| %s  |%s |%0.5f' % ( \
                            param_group['lr'], epoch,
                            loss_seg.data.cpu().numpy(),
                           v, dsc, dsc_all))
            meandsc = np.mean(dsc_mean)
            print(meandsc)

            if meandsc >= dsc_sav:
                torch.save(model_S.state_dict(), './only_7T/only7T2/val_data2017/fold' + str(fold) + '/_%s_%s_%s.pth' % (
                str(epoch).zfill(5), str(meandsc)[:8], 'WFNet'))
                dsc_sav = meandsc

