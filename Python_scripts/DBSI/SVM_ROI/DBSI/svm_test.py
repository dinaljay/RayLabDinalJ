
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import metrics

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

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data/all_patients_all_features_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop(['Group', 'Group_ID', 'dti_adc', 'dti_axial', 'dti_fa', 'dti_radial'], axis=1)
y = all_data['Group_ID']

# Scale data

from sklearn import preprocessing

X_scaled = preprocessing.scale(X)

#Implement leave one out cross validation
y_pred = []
auc = []
fpr = []
tpr = []
for i in range(len(X_scaled)):

    # Splitting Data for tuning hyerparameters
    X_train = np.delete(X_scaled, [i], axis=0)
    y_train = y.drop([i], axis=0)

    X_test = X_scaled[i]
    X_test = X_test.reshape(1, -1)
    y_test = y[i]

    # Tuning hyperparameters
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

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
    clf = SVC(C=cost, kernel="linear", probability=True)

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    # predict probabilities
    lr_probs = clf.predict_proba(X_test)

    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    # calculate scores
    auc.append(metrics.roc_auc_score(y_test, lr_probs))

    #calculate fpr and tpr
    fpr_val, tpr_val, _ = metrics.roc_curve(y_test, lr_probs)

    fpr.append(fpr_val)
    tpr.append(tpr_val)

    #Predict the response for test dataset
    y_pred.append(clf.predict(X_test))


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from statistics import mean

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y, np.asarray(y_pred)))

# Model Precision
print("Precision:", metrics.precision_score(y, np.asarray(y_pred)))

# Model Recall
print("Recall:", metrics.recall_score(y, np.asarray(y_pred)))

from sklearn.metrics import classification_report, confusion_matrix
cm1 = confusion_matrix(y, np.asarray(y_pred))

print("Confusion matrix: \n", cm1)
#print(classification_report(y, np.asarray(y_pred)))

## Caclulate number of true positives, true negatives, false negatives and false positives
total1=sum(sum(cm1))

# Accuracy
accuracy1=(cm1[0,0]+cm1[1,1])/total1
print('Accuracy:', accuracy1)

#Sensitivity or true positive rate
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity:', sensitivity1)

#Specificity or true negative rate
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity:', specificity1)

#Calculate AUC
#fpr, tpr, threshold = metrics.roc_curve(y, np.asarray(y_pred))
#roc_auc = metrics.auc(fpr, tpr)
roc_auc = mean(auc)
print("AUC:", roc_auc)

#Plot ROC curve

fpr = np.asarray(fpr)
tpr = np.asarray(tpr)

lw=2
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'darkorange', lw=lw, label = 'ROC curve (area = %0.2f)' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],color='navy', lw=lw, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()