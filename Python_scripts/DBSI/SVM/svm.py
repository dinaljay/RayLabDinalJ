
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

## Initialize features

dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "fiber1_axial_map", "fiber1_fa_map",
    "fiber1_radial_map", "fiber_fraction_map", "hindered_fraction_map", "restricted_fraction_map",
                "water_fraction_map", "axon_volume", "inflammation_volume"]

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


## Support Vector Machine
# Splitting Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_dbsi, all_ids, test_size=0.3,random_state=100) # 70% training and 30% test

# Cross Validation

# Generating model

from sklearn import svm

kernel = "linear"
clf = svm.SVC(kernel=kernel) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))