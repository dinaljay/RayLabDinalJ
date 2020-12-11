import numpy as np

dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "fiber1_axial_map", "fiber1_fa_map",
    "fiber1_radial_map", "fiber_fraction_map", "hindered_fraction_map", "restricted_fraction_map",
                "water_fraction_map", "axon_volume", "inflammation_volume"]

controls = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])
mild_cm_subjects = np.array([1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46])
moderate_cm_subjects = np.array([5,6,9,12,13,14,20,22,25,27,30,34,37,41,47])

all_cm = np.concatenate((mild_cm_subjects,moderate_cm_subjects),axis=0)

control_ids = np.array([0]*len(controls))
csm_ids = np.array([1]*len(all_cm))

all_ids = np.concatenate((control_ids,csm_ids),axis=0)

## Load Data

file1 = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data/CSM/csm_dti_adc_map_data.csv"
csm_dti_adc = np.genfromtxt(file1,delimiter=",",skip_header=1)

file2 = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data/Control/control_dti_adc_map_data.csv"
control_dti_adc = np.genfromtxt(file2,delimiter=",",skip_header=1)

all_dti_adc = np.concatenate((control_dti_adc,csm_dti_adc),axis=0)

## Support Vector Machine
# Splitting Data

from sklearn.model_selection import train_test_split
sample = all_dti_adc[:,1]
X_train, X_test, y_train, y_test = train_test_split(sample, all_ids, test_size=0.3,random_state=100) # 70% training and 30% test

control_sample = control_dti_adc[:,2]
print(sample)
# Generating model

from sklearn import svm

kernel = "linear"
clf = svm.SVC(kernel=kernel) # Linear Kernel