import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys

## Initialize features

radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map"]

improv_features = ["ndi_improve", "dash_improve", "mjoa_improve", "MCS_improve", "PCS_improve"]

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data

all_data = all_data_raw[radiographic_features]

#Set data to variables

X = all_data
y = all_data_raw['MCS_improve']

#Scale data

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
selector = RFE(estimator=svc, step=1, n_features_to_select=1)
final = selector.fit(X_scaled, y)

#print(final.support_)
#print("\n")
#print(final.ranking_)

print("Optimal number of features : %d" % selector.n_features_)

#print(X.columns)

#for i in range(X.shape[1]):
#    print('Column: %d, Selected %s, Rank: %.3f' % (i, selector.support_[i], selector.ranking_[i]))

features = list(X.columns)

d = {'Feature': features, 'Ranking': selector.ranking_}
rankings = pd.DataFrame(data=d)
rankings = rankings.sort_values(by=['Ranking'])
print(rankings)

sys.exit()
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (no of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()



