import numpy as np

def readtxt(path):
    file = open(path,'r')
    content = file.read()
    return content

Group      = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']    # 'Group1', 'Group2', 'Group3', 'Group4', 'Group5'
  
# model_name = 'DIRAC'                              
# model_name = 'cLapIRN_CM_tumorcore'
# model_name = 'cLapIRN_CM_tumor_2'
# model_name = 'cLapIRN_CAM_1stage_inverse1.0_fullmask'
# model_name = 'cLapIRN_CAM_1stage_inverse1.0_fullmask_aff'         
# model_name = 'cLapIRN_CAM_1stage_inverse1.0_fullmask_aff_diff_unp_128'
model_name = 'cLapIRN_CAM_1stage_inverse1.0_smooth1.0_fullmask_aff_diff_unpthre_128_camtemplate'
# model_name = 'cLapIRN_NCCMsk_aff'
# model_name = 'cLapIRN_Grad-CAM_aff'
                                                            # []  'Group1', 'Group2', 'Group3', 'Group4', 'Group5'           

# tre_near.txt, tre_far.txt
# robustness_near.txt, robustness_far.txt
# file_name: nJadet_near.txt, nJadet_far.txt,

scores = []
for i in range(len(Group)):
    # datapath = f'Log/{Group[i]}/{model_name}/tre.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/robustness.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/nJadet.txt'

    datapath = f'Log/{Group[i]}/{model_name}/tre_near.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/robustness_near.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/nJadet_near.txt'

    # datapath = f'Log/{Group[i]}/{model_name}/tre_far.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/robustness_far.txt'  # tre nJadet
    # datapath = f'Log/{Group[i]}/{model_name}/nJadet_far.txt'

    scores.extend([float(item) for item in readtxt(datapath).split("\n")[:-1]])

    print(np.array([float(item) for item in readtxt(datapath).split("\n")[:-1]]).mean(), np.array([float(item) for item in readtxt(datapath).split("\n")[:-1]]).std())
scores = np.array(scores)
print(scores.mean(), scores.std())
