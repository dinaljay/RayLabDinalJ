
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
y = all_data_raw['PCS_improve']

# Scale data

from sklearn import preprocessing

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
    clf = SVC(C=cost, kernel="linear")

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    temp = clf.predict(X_test)
    y_pred.append(temp[0])

    #Get confidence scores
    temp = clf.decision_function(X_test)
    y_conf.append(temp[0])

y = np.asarray(y)
y_pred = np.asarray(y_pred)
y_conf = np.asarray(y_conf)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y, y_pred))

# Model Precision
print("Precision:", metrics.precision_score(y, y_pred))

# Model Recall
print("Recall:", metrics.recall_score(y, y_pred))

#Model F1 score
f1 = metrics.f1_score(y, y_pred)
print("F1 Score:", f1)

from sklearn.metrics import classification_report, confusion_matrix
cm1 = confusion_matrix(y, y_pred)

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
fpr, tpr, threshold = metrics.roc_curve(y, y_conf)
#roc_auc = metrics.auc(fpr, tpr)
roc_auc = metrics.roc_auc_score(y, y_conf)
print("AUC:", roc_auc)

# get importance
importance = clf.coef_
print(importance)

sys.exit()
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

sys.exit()

#Plot ROC curve

#plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color='darkorange', linewidth=2, label='Area = %0.2f' %roc_auc)
#plt.plot([0, 1], [0, 1],color='navy', linewidth=2, linestyle='-', label='No Skill')
plt.legend(loc='lower right', fontsize=10)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1-Specificity', fontsize=13)
plt.ylabel('Sensitivity', fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.show()

#sys.exit()

# Plot precision recall curve

lr_precision, lr_recall, _ = metrics.precision_recall_curve(y, y_conf)
prc_auc = metrics.auc(metrics.recall_score(y, y_pred), metrics.precision_score(y, y_pred))

plt.figure()
#plt.title('Precision-Recall Curve')
#plt.plot([0, 1], [0, 0], color='navy', linewidth=2, linestyle='-', label='No Skill')
plt.plot(lr_recall, lr_precision, color='darkorange', linestyle='-', linewidth=2, label='Area = %0.2f' %prc_auc)
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right", fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.show()

