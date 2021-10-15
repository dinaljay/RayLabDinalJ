import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn import metrics
from itertools import cycle

## Initialize features


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map"]
"""
clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total", "dash_total",
                     "mjoa_total", "mjoa_recovery", "PCS", "MCS", "post_ndi_total", "post_mdi_total", "post_mjoa_total", "post_PCS", "post_MCS",
                     "change_ndi", "change_mdi", "change_dash", "change_mjoa", "change_PCS", "change_MCS"]
"""
clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total", "dash_total",
                     "PCS", "MCS", "mjoa_total"]

improv_features = ['ndi_improve', 'dash_improve', 'mjoa_improve', 'MCS_improve', 'PCS_improve']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features + clinical_features
all_data = all_data_raw[all_features]

#Variables for ROC and PRC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()
prc_auc = dict()

for n in range(len(improv_features)):

    # Set data to variables
    X = all_data
    y = all_data_raw[improv_features[n]]

    #Scale data
    X_scaled = preprocessing.scale(X)

    print(improv_features[n])
    print("\n")

    #Implement leave one out cross validation
    y_pred = []
    y_conf = []

    for i in range(len(X_scaled)):
        # Splitting Data for tuning hyerparameters
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = y.drop([i], axis=0)

        X_test = X_scaled[i]
        X_test = X_test.reshape(1, -1)
        y_test = y[i]

        # Tuning hyperparameters
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
        clf.fit(X_train, y_train)
        params = clf.best_params_
        cost = params['C']

        # Splitting Data for model
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = y.drop([i], axis=0)

        X_test = X_scaled[i]
        X_test = X_test.reshape(1, -1)
        y_test = y[i]

        # Generating SVM model
        clf = SVC(C=cost, kernel="linear")

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        temp = clf.predict(X_test)
        y_pred.append(temp[0])

        # Get confidence scores
        temp = clf.decision_function(X_test)
        y_conf.append(temp[0])

    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    y_conf = np.asarray(y_conf)

    #Calculate AUC
    fpr[n], tpr[n], _ = metrics.roc_curve(y, y_conf)
    roc_auc[n] = metrics.auc(fpr[n], tpr[n])
    #roc_auc[n] = metrics.roc_auc_score(y, y_conf)

    precision[n], recall[n], _ = metrics.precision_recall_curve(y.ravel(), y_conf.ravel())
    prc_auc[n] = metrics.auc(recall[n], precision[n])

colors = cycle(['darkorange', 'red', 'green', 'navy', 'purple'])

#Plot ROC curve

for i, color in zip(range(len(improv_features)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Area = {1:0.2f}' ''.format(i, roc_auc[i]))
plt.legend(loc='lower right', fontsize=10)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-Specificity', fontsize=13)
plt.ylabel('Sensitivity', fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.show()

# Plot precision recall curve

for i, color in zip(range(len(improv_features)), colors):
    plt.plot(recall[i], precision[i], lw=2, color=color, linestyle='-',
             label='Area = {1:0.2f}' ''.format(i, prc_auc[i]))

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.show()




