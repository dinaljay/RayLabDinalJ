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
import scipy.stats as ss
from sklearn.utils import resample
from itertools import cycle

## Initialize features


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map"]
"""
clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total", "dash_total",
                     "mjoa_total", "mjoa_recovery", "PCS", "MCS", "post_ndi_total", "post_mdi_total", "post_mjoa_total", "post_PCS", "post_MCS",
                     "change_ndi", "change_mdi", "change_dash", "change_mjoa", "change_PCS", "change_MCS"]
"""
clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total", "dash_total",
                     "PCS", "MCS", "mjoa_total", "Elix_1", "Elix_2", "Elix_3", "Elix_4", "Elix_5", "smoking"]

#improv_features = ['ndi_improve', 'dash_improve', 'mjoa_improve_1', 'MCS_improve', 'PCS_improve', 'mdi_improve', 'mjoa_improve_2']

improv_features = ['mjoa_improve_2']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)
#all_data_raw = all_data_raw.loc[all_data_raw['Group_ID'] == 2]

# Filter Data
all_features = radiographic_features + clinical_features
all_data = all_data_raw[all_features]

def mean_CI(data):
    mean = np.mean(np.array(data))
    CI = ss.t.interval(
        alpha=0.95,
        df=len(data) - 1,
        loc=np.mean(data),
        scale=ss.sem(data)
    )
    lower = CI[0]
    upper = CI[1]

    return mean, lower, upper

def roc_bootstrap(bootstrap, y_true, y_pred, y_conf):
    sample_accuracy = []
    sample_precision = []
    sample_recall = []
    sample_f1_Score = []
    sample_auc = []
    obs_vals = []

    obs_vals.append(metrics.accuracy_score(y_true, y_pred))
    obs_vals.append(metrics.precision_score(y_true, y_pred))
    obs_vals.append(metrics.recall_score(y_true, y_pred))
    obs_vals.append(metrics.f1_score(y_true, y_pred))
    obs_vals.append(metrics.roc_auc_score(y_true, y_conf))

    for j in range(bootstrap):

        index = range(len(y_pred))
        indices = resample(index, replace=True, n_samples=int(len(y_pred)))

        sample_accuracy.append(metrics.accuracy_score(y_true[indices], y_pred[indices]))
        sample_precision.append(metrics.precision_score(y_true[indices], y_pred[indices]))
        sample_recall.append(metrics.recall_score(y_true[indices], y_pred[indices]))
        sample_f1_Score.append(metrics.f1_score(y_true[indices], y_pred[indices]))
        sample_auc.append(metrics.roc_auc_score(y_true[indices], y_conf[indices]))

    ### Calculate mean and 95% CI
    accuracies = mean_CI(sample_accuracy)
    precisions = mean_CI(sample_precision)
    recalls = mean_CI(sample_recall)
    f1_scores = mean_CI(sample_f1_Score)
    aucs = mean_CI(sample_auc)

    obs_vals = np.around(obs_vals, 3)
    accuracies = np.around(accuracies, 3)
    precisions = np.around(precisions, 3)
    recalls = np.around(recalls, 3)
    f1_scores = np.around(f1_scores, 3)
    aucs = np.around(aucs, 3)

    # save results into dataframe
    stat_roc_1 = pd.DataFrame(
        [accuracies, precisions, recalls, f1_scores, aucs],
        columns=['mean', '95% CI -', '95% CI +'],
        index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    )

    stat_roc_2 = pd.DataFrame([obs_vals],
                              index=['Observed Value'],
                              columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
                              )
    stat_roc_fin = pd.concat([stat_roc_2.transpose(), stat_roc_1], axis=1)
    # print(stat_roc)

    return stat_roc_fin

#Variables for ROC and PRC curves
fpr = dict()
tpr = dict()
roc_auc = dict()
precision = dict()
recall = dict()

for n in range(len(improv_features)):

    # Set data to variables
    X = all_data
    y = all_data_raw[improv_features[n]]

    #Scale data
    X_scaled = preprocessing.scale(X)
    y = np.asarray(y)

    print(improv_features[n])

    #Implement leave one out cross validation
    y_pred = []
    y_conf = []

    for i in range(len(X_scaled)):
        # Splitting Data for tuning hyerparameters
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = np.delete(y, [i])

        X_test = X_scaled[i]
        X_test = X_test.reshape(1, -1)
        y_test = y[i]

        # Tuning hyperparameters
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
        clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
        clf.fit(X_train, y_train)
        params = clf.best_params_
        cost = params['C']

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

    final_result = roc_bootstrap(1000, y, y_pred, y_conf)
    print(final_result)





