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
import scipy.stats as st
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

#improv_features = ['ndi_improve', 'dash_improve', 'mjoa_improve', 'MCS_improve', 'PCS_improve']

improv_features = ['mjoa_improve']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features + clinical_features
all_data = all_data_raw[all_features]

accuracy_all = []
auc_all = []
recall_all = []
precision_all = []
f1_score_all = []

for k in range(5):

    # Variables for ROC and PRC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for n in range(len(improv_features)):

        # Set data to variables
        X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
        y = all_data_raw[improv_features[n]]

        #Scale data
        X_scaled = preprocessing.scale(X)

        # Tuning hyperparameters
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
        clf.fit(X_scaled, y)
        params = clf.best_params_
        cost = params['C']

        #RFE
        svc = SVC(kernel="linear", C=cost)
        selector = RFE(estimator=svc, step=1, n_features_to_select=10)
        final = selector.fit(X_scaled, y)

        features = list(X.columns)

        d = {'Feature': features, 'Ranking': selector.ranking_}
        rankings = pd.DataFrame(data=d)
        rankings = rankings.sort_values(by=['Ranking'])

        #Create list of rfe_features
        rfe_features = rankings["Feature"].tolist()
        rfe_features = rfe_features[0:10]

        print(improv_features[n])

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

        # Model Accuracy
        accuracy_all.append(metrics.accuracy_score(y, y_pred))

        # Model Precision
        precision_all.append(metrics.precision_score(y, y_pred))

        # Model Recall
        recall_all.append(metrics.recall_score(y, y_pred))

        # Model F1 score
        f1_score_all.append(metrics.f1_score(y, y_pred))

        #Calculate AUC
        fpr[n], tpr[n], _ = metrics.roc_curve(y, y_conf)
        roc_auc[n] = metrics.auc(fpr[n], tpr[n])
        auc_all.append(roc_auc[n])



# Accuracy
accuracy_fin = np.asarray(accuracy_all)
print(accuracy_all)
print("Mean Accuracy:", np.mean(accuracy_all))
print("95% CI", st.t.interval(alpha=0.95, df=len(accuracy_fin)-1, loc=np.mean(accuracy_fin), scale=st.sem(accuracy_fin)))
print("\n")

# Precision
precision_fin = np.asarray(precision_all)
print("Mean Precision:", np.mean(precision_all))
print("95% CI", st.t.interval(alpha=0.95, df=len(precision_fin)-1, loc=np.mean(precision_fin), scale=st.sem(precision_fin)))
print("\n")

# Recall
recall_fin = np.mean(recall_all)
print("Mean Recall:", np.mean(recall_all))
print("95% CI", st.t.interval(alpha=0.95, df=len(recall_fin)-1, loc=np.mean(recall_fin), scale=st.sem(recall_fin)))
print("\n")

# F1 score
f1_score_fin = np.asarray(f1_score_all)
print("Mean F1 Score:", np.mean(f1_score_all))
print("95% CI", st.t.interval(alpha=0.95, df=len(f1_score_fin)-1, loc=np.mean(f1_score_fin), scale=st.sem(f1_score_fin)))
print("\n")

# AUC
auc_fin = np.asarray(auc_all)
print("Mean AUC:", np.mean(auc_all))
print("95% CI", st.t.interval(alpha=0.95, df=len(auc_fin)-1, loc=np.mean(auc_fin), scale=st.sem(auc_fin)))
print("\n")