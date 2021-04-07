
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

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_improv_vs_nonimprov/all_patients_all_features_data.csv'

all_data = pd.read_csv(url, header=0)

#X = all_data.drop(['Patient_ID', 'Group', 'Group_ID', 'dti_adc', 'dti_axial', 'dti_fa', 'dti_radial'], axis=1)
X = all_data.drop(['Patient_ID', 'Group', 'Group_ID'], axis=1)

#X = all_data.drop(['Patient_ID', 'Group', 'Group_ID'], axis=1)
y = all_data['Group_ID']

# Scale data

from sklearn import preprocessing

X_scaled = preprocessing.scale(X)

# Tuning hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
clf.fit(X_scaled, y)
params = clf.best_params_
cost = params['C']

## Recursive Feature Elimination

from sklearn.feature_selection import RFE

svc = SVC(kernel="linear", C=cost)
selector = RFE(estimator=svc, step=1)
final = selector.fit(X_scaled, y)

#print(final.support_)
#print("\n")
#print(final.ranking_)

print("Optimal number of features : %d" % selector.n_features_)

print(X.columns)

for i in range(X.shape[1]):
    print('Column: %d, Selected %s, Rank: %.3f' % (i, selector.support_[i], selector.ranking_[i]))


sys.exit()
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (no of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()



