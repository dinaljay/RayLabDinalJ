
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys

## Initialize features

dhi_features = ["b0_map", "dti_adc_map","dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_dirz_map", "dti_fa_map",
                "dti_g_map", "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_axial_map", "fiber1_dirx_map",
                "fiber1_diry_map", "fiber1_dirz_map", "fiber1_fa_map", "fiber1_fiber_fraction_map", "fiber1_radial_map", "fiber1_rgba_map",
                "fiber1_rgba_map_itk", "fiber2_axial_map", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_fa_map",
                "fiber2_fiber_fraction_map", "fiber2_radial_map", "fiber_fraction_map", "fraction_rgba_map", "hindered_adc_map",
                "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map",
                "water_fraction_map"]

dbsi_ia_features = ["b0_map", "dti_adc_map", "dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_fa_map", "dti_g_map",
                    "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_dirx_map", "fiber1_diry_map", "fiber1_dirz_map",
                    "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",
                    "fiber1_intra_radial_map", "fiber1_rgba_map_itk", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_extra_axial_map",
                    "fiber2_extra_fraction_map", "fiber2_extra_radial_map", "fiber2_intra_axial_map", "fiber2_intra_fraction_map", "fiber2_intra_radial_map",
                    "fraction_rgba_map", "hindered_adc_map", "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map",
                    "water_adc_map", "water_fraction_map"]

filter_dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map", "fiber1_fa_map",
                       "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                       "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map", "water_fraction_map"]

filter_dbsi_ia_features = ["fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",
                           "fiber1_intra_radial_map"]

rfe_features = ["fiber1_extra_axial_map", "restricted_adc_map", "dash_total", "fiber1_extra_fraction_map", "PCS",
                "fiber1_extra_radial_map", "fiber_fraction_map", "iso_adc_map", "water_fraction_map", "bmi"]

## Load Data
#DHI Data
url_dhi = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_by_CSM_group_data.csv'
all_data_dhi = pd.read_csv(url_dhi, header=0)

#DBSI-IA Data
url_dbsi_ia = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_by_CSM_group_data.csv'
all_data_dbsi_ia = pd.read_csv(url_dbsi_ia, header=0)

#Clinical Data
url_clinical = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/Clinical_data/csm_clinical_pre_all.csv'
all_data_clinical = pd.read_csv(url_clinical, header=0)

all_data_clinical = all_data_clinical.drop(['record_id', 'Group_ID', 'mJOA_ID'], axis=1)

# Filter Data
filter_dhi = all_data_dhi[filter_dhi_features]
filter_dbsi_ia = all_data_dbsi_ia[filter_dbsi_ia_features]

all_data = pd.concat([filter_dhi, filter_dbsi_ia, all_data_clinical], axis=1)
all_data = all_data[rfe_features]

#Set variables to data

X = all_data
#X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
y = all_data_dhi['Group_ID']
patient_count = X.shape[0]

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

y = label_binarize(y=y, classes=[0, 1, 2])
n_classes = y.shape[1]
#sys.exit()
# Scale data

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut

X_scaled = preprocessing.scale(X)

#Implement leave one out cross validation
y_true, y_pred, y_conf, y_score = [], [], [], []
cv = LeaveOneOut()

for train_i, test_i in cv.split(X_scaled):

    # Splitting Data for tuning hyerparameters
    X_train, X_test = X_scaled[train_i, :], X_scaled[test_i, :]
    y_train, y_test = y[train_i], y[test_i]

    # Tuning hyperparameters
    tuned_parameters = [{'estimator__kernel': ['linear'], 'estimator__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    clf = GridSearchCV(OneVsRestClassifier(SVC()), tuned_parameters, scoring='accuracy')
    clf.fit(X_train, y_train)
    params = clf.best_params_
    cost = params['estimator__C']

    # Generating SVM model
    clf = OneVsRestClassifier(SVC(C=cost, kernel="linear", probability=True))

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred.append(clf.predict(X_test))
    y_true.append(y_test)

    #Get confidence scores
    y_conf.append(clf.decision_function(X_test))

#sys.exit()
y = np.asarray(y)
y_true = np.reshape(np.asarray(y_true), (patient_count, n_classes))
y_pred = np.reshape(np.asarray(y_pred), (patient_count, n_classes))
y_conf = np.reshape(np.asarray(y_conf), (patient_count, n_classes))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from itertools import cycle

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_true, np.asarray(y_pred)))

# Model Precision
print("Precision:", metrics.precision_score(y_true, np.asarray(y_pred), average='weighted'))

# Model Recall
print("Recall:", metrics.recall_score(y_true, np.asarray(y_pred), average='weighted'))

#Model F1 score
f1 = metrics.f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)

#Calculate AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_conf[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_conf.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# Aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr

roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

micro_roc_auc = metrics.roc_auc_score(y_true, y_conf, multi_class="ovr", average="micro")
macro_roc_auc = metrics.roc_auc_score(y_true, y_conf, multi_class="ovr", average="macro")

print("Micro average ROC curve area:", micro_roc_auc)
print("Macro average ROC curve area:", macro_roc_auc)

#sys.exit()

# ROC curve

plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='purple', linestyle='-', linewidth=2)

plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='brown', linestyle='-', linewidth=2)

colors = cycle(['darkorange', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
plt.xlim([0, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for all patient groups')
plt.legend(loc="lower right")
plt.show()

#Precision Recall curve
plt.figure()
precision = dict()
recall = dict()
prc_auc = dict()

precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(), y_conf.ravel())
#prc_auc["micro"] = metrics.average_precision_score(y_true.ravel(), y_conf.ravel())
prc_auc["micro"] = metrics.auc(recall["micro"], precision["micro"])


colors = cycle(['darkorange', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i], y_conf[:, i])
    #prc_auc[i] = metrics.average_precision_score(y_true[:, i], y_conf[:, i])
    prc_auc[i] = metrics.auc(recall[i], precision[i])
    plt.plot(recall[i], precision[i], lw=2, color=color, linestyle='-',
             label='Precision-recall curve for class {0}(area = {1:0.2f})' ''.format(i, prc_auc[i]))

plt.plot(precision["micro"], recall["micro"], label='Micro-average precision-recall curve(area = {0:0.2f})'''.format(prc_auc["micro"]),
         color='purple', linestyle='-', linewidth=2)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.title("Precision-Recall curve")
plt.show()
