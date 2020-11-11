
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

## Initialize features

dhi_features = ["b0_map","dti_adc_map","dti_axial_map","dti_fa_map","fiber1_axial_map","fiber1_fa_map",\
    "fiber1_radial_map","fiber_fraction_map","hindered_fraction_map","restricted_fraction_map","water_fraction_map"]

controls = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])
mild_cm_subjects = np.array([1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46])
moderate_cm_subjects = np.array([5,6,9,12,13,14,20,22,25,27,30,34,35,37,39,41,47])

all_cm = np.concatenate((mild_cm_subjects,moderate_cm_subjects),axis=0)

control_ids = np.array([0]*len(controls))
csm_ids = np.array([1]*len(all_cm))

all_ids = np.concatenate((control_ids,csm_ids),axis=0)

## Load Data

for i in range(len(dhi_features)):
    feature = dhi_features[i]
    file_in = "csm_"+feature+"_data.csv"
    file1 = path.join("/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data/CSM",file_in)
    csm_dbsi = np.genfromtxt(file1,delimiter=",",skip_header=1)

    file_in = "control_"+feature+"_data.csv"
    file2 = path.join("/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data/Control",file_in)
    control_dbsi = np.genfromtxt(file2,delimiter=",",skip_header=1)

    temp = np.concatenate((control_dbsi,csm_dbsi),axis=0)

    if (i==0):
        all_dbsi = temp
    else:
        all_dbsi = np.concatenate((all_dbsi, temp), axis=1)

## Recursive Feature Elimination

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.svm import SVC

X = all_dbsi
y = all_ids

#estimator = SVR(kernel="linear")
svc = SVC(kernel="rbf",C=1,gamma="auto")
selector = RFECV(estimator=svc, step=1, cv=4)
#final = selector.fit(X, y)

#print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

#print(final.support_)
#print("\n")
#print(final.ranking_)

