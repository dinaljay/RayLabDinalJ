#This script does a grid search method with 4 fold cross validation to determine
# the best cost and gamme values

import numpy as np
import os.path as path
import sys
import matplotlib.pyplot as plt

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


sys.exit()
## Grid Search

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

X = all_dbsi
y = all_ids

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                     'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    {'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 8, 10]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
