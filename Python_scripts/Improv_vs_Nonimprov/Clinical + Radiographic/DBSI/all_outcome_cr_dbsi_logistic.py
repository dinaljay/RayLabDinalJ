import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from itertools import cycle

## Initialize features


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map", "fiber1_fa_map",
                         "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                         "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map",
                         "water_fraction_map", "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map",
                         "fiber1_intra_axial_map", "fiber1_intra_fraction_map", "fiber1_intra_radial_map"]
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
    X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
    y = all_data_raw[improv_features[n]]

    #Scale data
    X_scaled = preprocessing.scale(X)

    #RFE
    logref = LogisticRegression(random_state=0)
    selector = RFE(estimator=logref, step=1, n_features_to_select=1)
    final = selector.fit(X_scaled, y)

    features = list(X.columns)

    d = {'Feature': features, 'Ranking': selector.ranking_}
    rankings = pd.DataFrame(data=d)
    rankings = rankings.sort_values(by=['Ranking'])

    #Create list of rfe_features
    rfe_features = rankings["Feature"].tolist()
    rfe_features = rfe_features[0:10]

    print(improv_features[n])
    print(rfe_features)

    #Set data to variables
    X = all_data[rfe_features]
    del rfe_features
    # Scale data
    X_scaled = preprocessing.scale(X)

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

        # Splitting Data for model
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = y.drop([i], axis=0)

        X_test = X_scaled[i]
        X_test = X_test.reshape(1, -1)
        y_test = y[i]

        # Generating SVM model
        clf = LogisticRegression(random_state=0)

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

    # Model Accuracy
    print("Accuracy:", metrics.accuracy_score(y, y_pred))

    # Model Precision
    print("Precision:", metrics.precision_score(y, y_pred))

    # Model Recall
    print("Recall:", metrics.recall_score(y, y_pred))

    # Model F1 score
    print("F1 Score:", metrics.f1_score(y, y_pred))

    #Calculate AUC
    fpr[n], tpr[n], _ = metrics.roc_curve(y, y_conf)
    roc_auc[n] = metrics.auc(fpr[n], tpr[n])
    #roc_auc[n] = metrics.roc_auc_score(y, y_conf)
    print("AUC:", roc_auc[n])
    print("\n")

