import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import csv
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from scipy.ndimage.interpolation import map_coordinates

from Functions import Validation_Brats, generate_grid_unit, save_img
from bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1,\
                                 laplacian_lvl2,\
                                 laplacian_lvl3,\
                                 SpatialTransform_unit

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

  

def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,)+ X.shape)
    return X

def load_5D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,)+(1,)+ X.shape)
    return X

def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img

def read_landmarks(csvpath):
    file = open(csvpath)
    f_reader = csv.reader(file)
    next(f_reader)

    landmarks = []
    for f_line in f_reader:
        landmarks.append([float(f_line[1]), float(f_line[2])+239, float(f_line[3])])

    file.close()
    return np.array(landmarks)

def load_landmarks(mov_csv_name, fixed_csv_name):
    fixed_csv_file = open(fixed_csv_name)
    fixed_csv_reader = csv.reader(fixed_csv_file)
    next(fixed_csv_reader)

    mov_csv_file = open(mov_csv_name)
    mov_csv_reader = csv.reader(mov_csv_file)
    next(mov_csv_reader)
    
    fixed_key_list = []
    mov_key_list = []

    for fixed_csv_line, mov_csv_line in zip(fixed_csv_reader, mov_csv_reader):
        fixed_key_list.append([float(fixed_csv_line[1]), float(fixed_csv_line[2])+239., float(fixed_csv_line[3])])
        mov_key_list.append([float(mov_csv_line[1]), float(mov_csv_line[2])+239., float(mov_csv_line[3])])

    fixed_csv_file.close()
    mov_csv_file.close()
    return np.array(mov_key_list), np.array(fixed_key_list)

def compute_tre(x, y, spacing=(1, 1, 1)):
    return np.linalg.norm((x - y) * spacing, axis=1)

def compute_Robustness(tre_bsline, tre_logits):
    diff = tre_logits -tre_bsline
    return np.where(diff < 0., 1., 0.).mean()

def compute_Ja_less_0(displacement, grid_unit, roi=None):

    Jdet = Get_Jac(displacement, grid_unit)
    # if roi==None:
    #     out = np.where(Jdet > 0., 0., 1.).mean()
    # else:
    #     out = np.where(Jdet > 0., 0., 1.)
    #     out = (out * roi).sum() / roi.sum()
    roi = roi[0, :, :-1, :-1, :-1]
    out = np.where(Jdet > 0., 0., 1.)
    out = (out * roi).sum() / roi.sum()
    return out
 
def Get_Jac(displacement, sample_grid):
    J = displacement + sample_grid

    dx = J[:, :, 1:, :-1, :-1] - J[:, :, :-1, :-1, :-1]
    dy = J[:, :, :-1, 1:, :-1] - J[:, :, :-1, :-1, :-1]
    dz = J[:, :, :-1, :-1, 1:] - J[:, :, :-1, :-1, :-1]

    Jdet0 = dx[:,2,:,:,:] * (dy[:,1,:,:,:] * dz[:,0,:,:,:] - dy[:,0,:,:,:] * dz[:,1,:,:,:])
    Jdet1 = dy[:,2,:,:,:] * (dx[:,0,:,:,:] * dz[:,1,:,:,:] - dx[:,1,:,:,:] * dz[:,0,:,:,:])
    Jdet2 = dz[:,2,:,:,:] * (dy[:,0,:,:,:] * dx[:,1,:,:,:] - dy[:,1,:,:,:] * dx[:,0,:,:,:])

    return Jdet0 + Jdet1 + Jdet2

def write_log(log_dir, value_list, test_group):
    with open(log_dir, "w") as log:
        # log.write(test_group + "\n")
        [log.write(str(item) + "\n") for item in value_list]
        log.close()

def test():
    print("Testing model...")
    # model
    # -----------------------------------------------------------------------------
    model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(2, 3, start_channel, is_train=True, diff=True, 
                                                                   imgshape=imgshape_4,
                                                                   range_flow=range_flow, num_block=num_cblock).cuda()
    model_lvl2 = laplacian_lvl2(2, 3, start_channel, is_train=True, diff=True,
                                                                   imgshape=imgshape_2,
                                                                   range_flow=range_flow, model_lvl1=model_lvl1,
                                                                   num_block=num_cblock).cuda()

    model = laplacian_lvl3(2, 3, start_channel, is_train=True, diff=True, imgshape=imgshape,
                               range_flow=range_flow, model_lvl2=model_lvl2,
                               num_block=num_cblock).cuda()

    model_path = model_name
    model.load_state_dict(torch.load(model_path))

    transform = SpatialTransform_unit().cuda()
    # transform_nearest = SpatialTransformNearest_unit().cuda()
    # diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # Validation
    val_fixed_list  = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    val__list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))

    # landmarks
    #----------------------------------------------------------------------------------------------------------------
    # within 30mm
    val_fixed_csv_list  = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarkswithin30mm.csv"))
    val_moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarkswithin30mm.csv"))
    val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])
    val__csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarkswithin30mm.csv"))

    # # outside 30mm
    # val_fixed_csv_list  = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarksoutside30mm.csv"))
    # val_moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarksoutside30mm.csv"))
    # val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])
    # val__csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarksoutside30mm.csv"))

    # # all
    # val_fixed_csv_list  = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarks.csv"))
    # val_moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv"))
    # val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])
    # val__csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv"))

    # tumor img
    val_fixed_tumor_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*0000*tumorcore.nii.gz"))

    # dilated tumor img
    val_fixed_dilated_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*0000*tumordilated.nii.gz"))

    # brain mask img
    val_fixed_mask_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*0000_brainmask.nii.gz"))
    print(len(val_fixed_tumor_list), len(val_fixed_dilated_list), len(val_fixed_mask_list))

    save_path = '../Result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    template = nib.load(val_fixed_list[0])
    header, ine = template.header, template.ine

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    tre_total, robustness_total, nja_total, nja_near_total, nja_far_total = [], [], [], [], []
    print("\nValiding...")

    if nJadet_dir != None:
        for batch_idx, val_fixed_line in enumerate(val_fixed_list):
            Y_ori = torch.from_numpy(imgnorm(load_5D(val__list[batch_idx]))).float().to(device)
            X_ori = torch.from_numpy(imgnorm(load_5D(val_fixed_list[batch_idx]))).float().to(device)
            
            X_label     = read_landmarks(val__csv_list[batch_idx])
            X_label_ori = read_landmarks(val_moving_csv_list[batch_idx])
            Y_label     = read_landmarks(val_fixed_csv_list[batch_idx])

            X_dilated, X_tumorcore, X_brainmask =  load_5D(val_fixed_dilated_list[batch_idx]), load_5D(val_fixed_tumor_list[batch_idx]), load_5D(val_fixed_mask_list[batch_idx])

            ori_img_shape = X_ori.shape[2:]
            h, w, d = ori_img_shape

            X = F.interpolate(X_ori, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y_ori, size=imgshape, mode='trilinear')

            with torch.no_grad():
                reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
                # F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, fea_xy = model(X, Y, reg_code)
                # F_Y_X, Y_X, X_4y, F_yx, F_yx_lvl1, F_yx_lvl2, _, fea_yx = model(Y, X, reg_code)
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, _, _ = model(X, Y, reg_code)            # fixed --> moving   术前 --> 术后
                F_Y_X, Y_X, X_4y, F_yx, F_yx_lvl1, F_yx_lvl2, _, _, _ = model(Y, X, reg_code)            # moving --> fixed   术后 --> 术前

            
                F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)
                F_Y_X = F.interpolate(F_Y_X, size=ori_img_shape, mode='trilinear', align_corners=True)

                grid_unit = generate_grid_unit(ori_img_shape)
                grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()
            
                # X_Y = transform(X_ori, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)
                # Y_X = transform(Y_ori, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)

                # save_img(X_Y.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_X_Y.nii.gz", header=header, ine=ine)
                # save_img(Y_X.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_Y_X.nii.gz", header=header, ine=ine)

                if Y_label.shape[0] != 0:

                    full_F_X_Y = torch.zeros(F_X_Y.shape)
                    full_F_X_Y[0, 0] = F_X_Y[0, 2] * (h - 1) / 2
                    full_F_X_Y[0, 1] = F_X_Y[0, 1] * (w - 1) / 2
                    full_F_X_Y[0, 2] = F_X_Y[0, 0] * (d - 1) / 2

                    # TRE
                    full_F_X_Y = full_F_X_Y.cpu().numpy()[0]

                    fixed_keypoints = Y_label
                    moving_keypoints = X_label

                    moving_disp_x = map_coordinates(full_F_X_Y[0], moving_keypoints.transpose())
                    moving_disp_y = map_coordinates(full_F_X_Y[1], moving_keypoints.transpose())
                    moving_disp_z = map_coordinates(full_F_X_Y[2], moving_keypoints.transpose())
                    lms_moving_disp = np.array((moving_disp_x, moving_disp_y, moving_disp_z)).transpose()

                    warped_moving_keypoint = moving_keypoints + lms_moving_disp

                    tre_scores = compute_tre(warped_moving_keypoint, fixed_keypoints, spacing=(1., 1., 1.))
                    tre_score = tre_scores.mean()
                    tre_total.append(tre_score)

                    robustness_score = compute_Robustness(compute_tre(Y_label, X_label_ori, spacing=(1., 1., 1.)), tre_scores)
                    robustness_total.append(robustness_score)

                    print(batch_idx, ": TRE: ", tre_score,  ": Robustness: ", robustness_score)

            
                F_Y_X = F_Y_X.cpu().numpy()
                grid_unit = grid_unit.permute(0, 4, 1, 2, 3).cpu().numpy()

                nja_score = compute_Ja_less_0(F_Y_X, grid_unit, X_brainmask)
                nja_total.append(nja_score)

                nja_score_far  = compute_Ja_less_0(F_Y_X, grid_unit, (1. - X_dilated) * X_brainmask)
                nja_far_total.append(nja_score_far)

                if X_dilated.sum() != 0:
                    # nja_score_near = compute_Ja_less_0(F_Y_X, grid_unit, X_dilated)
                    nja_score_near = compute_Ja_less_0(F_Y_X, grid_unit, X_dilated - X_tumorcore)
                    nja_near_total.append(nja_score_near)

        write_log(tre_dir, tre_total, test_group)
        write_log(robustness_dir, robustness_total, test_group)

        write_log(nJadet_dir, nja_total, test_group)
        write_log(nJadet_near_dir, nja_near_total, test_group)
        write_log(nJadet_far_dir, nja_far_total, test_group)
    
    else:
        for batch_idx, val_fixed_line in enumerate(val_fixed_list):
            Y_ori = torch.from_numpy(imgnorm(load_5D(val__list[batch_idx]))).float().to(device)
            X_ori = torch.from_numpy(imgnorm(load_5D(val_fixed_list[batch_idx]))).float().to(device)
            
            X_label     = read_landmarks(val__csv_list[batch_idx])
            X_label_ori = read_landmarks(val_moving_csv_list[batch_idx])
            Y_label     = read_landmarks(val_fixed_csv_list[batch_idx])

            ori_img_shape = X_ori.shape[2:]
            h, w, d = ori_img_shape

            X = F.interpolate(X_ori, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y_ori, size=imgshape, mode='trilinear')

            with torch.no_grad():
                reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
                # F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, fea_xy = model(X, Y, reg_code)
                # F_Y_X, Y_X, X_4y, F_yx, F_yx_lvl1, F_yx_lvl2, _, fea_yx = model(Y, X, reg_code)
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, _, _ = model(X, Y, reg_code)            # fixed --> moving   术前 --> 术后
                F_Y_X, Y_X, X_4y, F_yx, F_yx_lvl1, F_yx_lvl2, _, _, _ = model(Y, X, reg_code)            # moving --> fixed   术后 --> 术前

                F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)
                F_Y_X = F.interpolate(F_Y_X, size=ori_img_shape, mode='trilinear', align_corners=True)

                grid_unit = generate_grid_unit(ori_img_shape)
                grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()
            
                # X_Y = transform(X_ori, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)
                # Y_X = transform(Y_ori, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)

                # save_img(X_Y.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_X_Y.nii.gz", header=header, ine=ine)
                # save_img(Y_X.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_Y_X.nii.gz", header=header, ine=ine)

                if Y_label.shape[0] != 0:

                    full_F_X_Y = torch.zeros(F_X_Y.shape)
                    full_F_X_Y[0, 0] = F_X_Y[0, 2] * (h - 1) / 2
                    full_F_X_Y[0, 1] = F_X_Y[0, 1] * (w - 1) / 2
                    full_F_X_Y[0, 2] = F_X_Y[0, 0] * (d - 1) / 2

                    # TRE
                    full_F_X_Y = full_F_X_Y.cpu().numpy()[0]

                    fixed_keypoints = Y_label
                    moving_keypoints = X_label

                    moving_disp_x = map_coordinates(full_F_X_Y[0], moving_keypoints.transpose())
                    moving_disp_y = map_coordinates(full_F_X_Y[1], moving_keypoints.transpose())
                    moving_disp_z = map_coordinates(full_F_X_Y[2], moving_keypoints.transpose())
                    lms_moving_disp = np.array((moving_disp_x, moving_disp_y, moving_disp_z)).transpose()

                    warped_moving_keypoint = moving_keypoints + lms_moving_disp

                    tre_scores = compute_tre(warped_moving_keypoint, fixed_keypoints, spacing=(1., 1., 1.))
                    tre_score = tre_scores.mean()
                    tre_total.append(tre_score)

                    robustness_score = compute_Robustness(compute_tre(Y_label, X_label_ori, spacing=(1., 1., 1.)), tre_scores)
                    robustness_total.append(robustness_score)

                    print(batch_idx, ": TRE: ", tre_score,  ": Robustness: ", robustness_score)


        write_log(tre_dir, tre_total, test_group)
        write_log(robustness_dir, robustness_total, test_group)
        
    tre_total = np.array(tre_total)
    nja_total = np.array(nja_total)
    print("TRE mean: ", tre_total.mean(), 'TRE std:', tre_total.std(), 'Nja mean:', nja_total.mean())
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--start_channel", type=int,
                        dest="start_channel",
                        default=6,  # default:8, 7 for stage
                        help="number of start channels")
    parser.add_argument("--num_cblock", type=int,
                        dest="num_cblock",
                        default=5,
                        help="Number of conditional block")
    # ------------------------------------------------------------------------------

    parser.add_argument("--train_datapath", type=str,
                        dest="train_datapath",
                        default='/data/wuchongwei/Data/Registration/Brain/BratsReg/Train-Groups',
                        help="data path for training images")
    parser.add_argument("--val_datapath", type=str,
                        dest="val_datapath",
                        default='/home/wuchongwei/DLCode/Data/Brain/BratsReg/Validation-Groups',
                        help="data path for training images")
    parser.add_argument("--test_datapath", type=str,
                        dest="test_datapath",
                        default='/data/wuchongwei/Data/Registration/Brain/BratsReg/Test-Groups',
                        help="data path for training images")


    opt = parser.parse_args()
    start_channel = opt.start_channel
    num_cblock = opt.num_cblock
    

    validation_groups = ['Group1', 'Group1', 'Group1', 'Group1', 'Group5']
    test_groups       = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
    # ------------------------------------------------------------------------------
    group_order = 5
    test_group  = test_groups[group_order-1]
                                  # 1:                                              
    model_path= 'code/Group_5_model.pth'
   
    
    

    # ----------------------------------------------------------------------------------------------------------------
    # within 30mm
    tre_dir = tre_near_dir = f'Log/{test_group}/{name}/tre_near.txt'
    robustness_dir = robustness_near_dir = f'Log/{test_group}/{name}/robustness_near.txt'
    nJadet_dir = None
    nJadet_near_dir = None
    nJadet_far_dir  = None

    # # outside 30mm
    # tre_dir = tre_far_dir  = f'Log/{test_group}/{name}/tre_far.txt'
    # robustness_dir = robustness_far_dir  = f'Log/{test_group}/{name}/robustness_far.txt'
    # nJadet_dir = None
    # nJadet_near_dir = None
    # nJadet_far_dir  = None

    # # all
    # tre_dir = f'Log/{test_group}/{name}/tre.txt'
    # robustness_dir = f'Log/{test_group}/{name}/robustness.txt'
    # nJadet_dir = f'Log/{test_group}/{name}/nJadet.txt'
    # nJadet_near_dir = f'Log/{test_group}/{name}/nJadet_near.txt'
    # nJadet_far_dir  = f'Log/{test_group}/{name}/nJadet_far.txt'

    
    opt = parser.parse_args()
    datapath = f"{opt.test_datapath}/{test_group}"
    model_name = model_path

    img_h, img_w, img_d = 128, 128, 80
    imgshape   = (img_h, img_w, img_d)
    imgshape_4 = (img_h//4, img_w//4, img_d//4)
    imgshape_2 = (img_h//2, img_w//2, img_d//2)

    range_flow = 0.4
    print("Testing %s ..." % model_name)
    test()
