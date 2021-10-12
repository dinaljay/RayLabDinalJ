
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_by_CSM_group_data.csv'
all_data = pd.read_csv(url, header=0)

X = all_data[['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map']]
y = all_data['Group_ID']
patient_count = X.shape[0]

# Scale data

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

colors = cycle(['darkorange', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    #plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Area = {1:0.2f}' ''.format(i, roc_auc[i]))

#plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]), color='purple', linestyle='-', linewidth=2)
plt.plot(fpr["micro"], tpr["micro"], label='Area = {0:0.2f}' ''.format(roc_auc["micro"]), color='purple', linestyle='-', linewidth=2)

"""
plt.plot(fpr["macro"], tpr["macro"],
         label='Macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='brown', linestyle='-', linewidth=2)
"""

#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-')
plt.xlim([0.05, 1.05])
plt.ylim([0.05, 1.05])
plt.xlabel('1-Specificity', fontsize=13)
plt.ylabel('Sensitivity', fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
#plt.title('ROC curve for all patient groups')
plt.legend(loc="lower right", fontsize=10)
plt.grid()
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
    #plt.plot(recall[i], precision[i], lw=2, color=color, linestyle='-', label='Precision-recall curve for class {0}(area = {1:0.2f})' ''.format(i, prc_auc[i]))
    plt.plot(recall[i], precision[i], lw=2, color=color, linestyle='-', label='Area = {1:0.2f}' ''.format(i, prc_auc[i]))

"""
plt.plot(precision["micro"], recall["micro"], label='Micro-average precision-recall curve(area = {0:0.2f})'''.format(prc_auc["micro"]),
         color='purple', linestyle='-', linewidth=2)
"""
plt.plot(precision["micro"], recall["micro"], label='Area = {0:0.2f}'''.format(prc_auc["micro"]),
         color='purple', linestyle='-', linewidth=2)

plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.xlim([0.05, 1.05])
plt.ylim([0.05, 1.05])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc="lower right", fontsize=10)
#plt.title("Precision-Recall curve")
plt.grid()
plt.show()

