import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates

# -------------------------------------------------------------------------------
# tre
# -------------------------------------------------------------------------------
def compute_tre(x, y, spacing=(1, 1, 1)):
    return np.linalg.norm((x - y) * spacing, axis=1)

def apply_tansform_to_points(forwarddisp, fixed_landmarks, imgshape):   
    h, w, d = imgshape

    # compute the displacement of the fixed landmarks
    fixed_disp_x = map_coordinates(forwarddisp[0, 2]*(h - 1)/2, fixed_landmarks.transpose())
    fixed_disp_y = map_coordinates(forwarddisp[0, 1]*(w - 1)/2, fixed_landmarks.transpose())
    fixed_disp_z = map_coordinates(forwarddisp[0, 0]*(d - 1)/2, fixed_landmarks.transpose())
    fixed_lms_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()

    # compute the transformed landmarks
    transformed_fixed_lms = fixed_landmarks + fixed_lms_disp
    return transformed_fixed_lms

def apply_tansform_to_points_(forwarddisp, fixed_landmarks, imgshape):   
    full_forwarddisp = torch.zeros(forwarddisp.shape)
    h, w, d = imgshape

    # compute the complete difplacement (offset) filed
    full_forwarddisp[0, 0] = forwarddisp[0, 2] * (h - 1) / 2
    full_forwarddisp[0, 1] = forwarddisp[0, 1] * (w - 1) / 2
    full_forwarddisp[0, 2] = forwarddisp[0, 0] * (d - 1) / 2

    full_forwarddisp = full_forwarddisp.cpu().numpy()[0]

    # compute the displacement of the fixed landmarks
    fixed_disp_x = map_coordinates(full_forwarddisp[0], fixed_landmarks.transpose())
    fixed_disp_y = map_coordinates(full_forwarddisp[1], fixed_landmarks.transpose())
    fixed_disp_z = map_coordinates(full_forwarddisp[2], fixed_landmarks.transpose())
    fixed_lms_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()

    # compute the transformed landmarks
    transformed_fixed_lms = fixed_landmarks + fixed_lms_disp
    return transformed_fixed_lms


# -------------------------------------------------------------------------------
# robustness
# -------------------------------------------------------------------------------
def compute_Robustness(tre_bsline, tre_logits):
    diff = tre_logits -tre_bsline
    return np.where(diff < 0, 1, 0).mean()


# -------------------------------------------------------------------------------
# robustness
# -------------------------------------------------------------------------------
def compute_Ja_less_0(displacement, grid_unit, roi=None, weighted=False):

    Jdet = Get_Jac(displacement, grid_unit)
    if weighted:
        out = np.where(Jdet > 0, 0, 1)
        out = (out * roi).sum() / roi.sum()
    else:
        out = np.where(Jdet > 0, 0, 1).mean()
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

    # dy = J[:, :, 1:, :-1, :-1] - J[:, :, :-1, :-1, :-1]
    # dx = J[:, :, :-1, 1:, :-1] - J[:, :, :-1, :-1, :-1]
    # dz = J[:, :, :-1, :-1, 1:] - J[:, :, :-1, :-1, :-1]

    # Jdet0 = dx[:,0,:,:,:] * (dy[:,1,:,:,:] * dz[:,2,:,:,:] - dy[:,2,:,:,:] * dz[:,1,:,:,:])
    # Jdet1 = dx[:,1,:,:,:] * (dy[:,0,:,:,:] * dz[:,2,:,:,:] - dy[:,2,:,:,:] * dz[:,0,:,:,:])
    # Jdet2 = dx[:,2,:,:,:] * (dy[:,0,:,:,:] * dz[:,1,:,:,:] - dy[:,1,:,:,:] * dz[:,0,:,:,:])

    # return Jdet0 - Jdet1 + Jdet2

 

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if i == 0:
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice/num_count

def dice_torch(moved, fixed):
    unique_class = torch.unique(fixed)
    dice = 0
    num_count = 0
    for i in unique_class:
        if i == 0:
            continue

        sub_dice = torch.sum(fixed[moved == i] == i) * 2.0 / (torch.sum(moved == i) + torch.sum(fixed == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice/num_count


def NCC_mask(fixed, moving, mask=None):
    # br_msk = 1.-np.where(fixed > 0., 0., 1.)*np.where(moving >0., 0., 1.)
    br_msk = np.where(fixed > 0., 1., 0.)

    if mask is not None:
        br_msk = mask * br_msk

    m_fixed  = np.ma.masked_array(fixed, mask=br_msk==0.)
    m_moving = np.ma.masked_array(moving, mask=br_msk==0.)

    f_mean = np.ma.mean(m_fixed)
    m_mean = np.ma.mean(m_moving)
    cov    = np.ma.mean((m_fixed-f_mean)*(m_moving-m_mean))
    ncc    = cov / (np.ma.std(m_fixed)*np.ma.std(m_moving))
    return ncc