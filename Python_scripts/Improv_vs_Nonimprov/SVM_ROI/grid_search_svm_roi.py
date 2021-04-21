#This script does a grid search method with 4 fold cross validation to determine
# the best cost and gamme values

import numpy as np
import os.path as path
import sys
import matplotlib.pyplot as plt
import pandas as pd

dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "fiber1_axial_map", "fiber1_fa_map",
    "fiber1_radial_map", "fiber_fraction_map", "hindered_fraction_map", "restricted_fraction_map",
                "water_fraction_map", "axon_volume", "inflammation_volume"]

controls = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])
mild_cm_subjects = np.array([1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46])
moderate_cm_subjects = np.array([5,6,9,12,13,14,20,22,25,27,30,34,37,41,47])

all_cm = np.concatenate((mild_cm_subjects,moderate_cm_subjects),axis=0)

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/Pycharm_Data_improv_vs_nonimprov/Pre_op/ROI/all_patients_all_features_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop(['Patient_ID', 'Group', 'Group_ID', 'dti_adc', 'dti_axial', 'dti_fa', 'dti_radial'], axis=1)
#X = all_data[['fiber_axial', 'axon_volume', 'fiber_radial', 'water_fraction']]
y = all_data['Group_ID']

## Grid Search

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e1, 10e2,10e3],
#                     'C': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10e1, 10e2, 10e3, 10e4, 10e5]},
#                    {'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
#                    {'kernel': ['poly'], 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     'gamma': [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e1, 10e2,10e3]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s' % score)
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
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
