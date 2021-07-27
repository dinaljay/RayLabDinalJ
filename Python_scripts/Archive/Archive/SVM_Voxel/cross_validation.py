
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

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data/all_patients_all_features_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop('Group', axis=1)
y = all_data['Group']

## Cross validation

from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

cv = LeaveOneOut()

print(type(cv.split(X)))
sys.exit()
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
    # split data
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]
    # fit model
    model = SVC(C=3.0, kernel="linear", gamma=0.001)
    model.fit(X_train, y_train)
    # evaluate model
    yhat = model.predict(X_test)
    # store
    y_true.append(y_test[0])
    y_pred.append(yhat[0])

# calculate accuracy
acc = accuracy_score(y_true, y_pred)
print('Accuracy: %.3f' % acc)

