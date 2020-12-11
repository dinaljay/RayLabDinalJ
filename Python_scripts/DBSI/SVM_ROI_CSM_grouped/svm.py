
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd

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

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data/all_patients_all_features_by_CSM_group_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop(['Patient_ID', 'Group', 'Group_ID'], axis=1)
y = all_data['Group_ID']

# Scale data

from sklearn import preprocessing

X_scaled = preprocessing.scale(X)

## Support Vector Machine
# Splitting Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=100) # 70% training and 30% test

# Cross Validation

# Generating model

from sklearn.svm import SVC

clf = SVC(C=7.0, kernel="rbf", gamma=0.0001)

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
