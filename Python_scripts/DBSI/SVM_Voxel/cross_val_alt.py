
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys

## Initialize features

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

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_Voxel/DBSI_CSV_Data/all_patients_all_features_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop('Group', axis=1)
y = all_data['Group']

## Cross validation

from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

cv = LeaveOneOut()

print(type(cv.split(X)))
#sys.exit()
    # split data

    # fit model
model = SVC(C=2.0, kernel="linear")
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# calculate accuracy
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))