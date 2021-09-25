
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

## Load Data

url_dhi = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_data.csv'
all_data_dhi = pd.read_csv(url_dhi, header=0)

url_dbsi_ia = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_data.csv'
all_data_dbsi_ia = pd.read_csv(url_dbsi_ia, header=0)

# Filter Data
filter_dhi = all_data_dhi[filter_dhi_features]
filter_dbsi_ia = all_data_dbsi_ia[filter_dbsi_ia_features]

all_data = pd.concat([filter_dhi, filter_dbsi_ia], axis=1)

#Set NaN data to 0

for col in all_data.columns:
    all_data[col] = all_data[col].fillna(0)

X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
y = all_data_dhi['Group_ID']

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

lw=2
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='SVM (area = %0.2f)' %roc_auc)
plt.plot([0, 1], [0, 1],color='navy', lw=lw, linestyle='--', label='No Skill')
plt.legend(loc='lower right')
plt.xlim([-0.1, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()

#sys.exit()

# Plot precision recall curve

lr_precision, lr_recall, _ = metrics.precision_recall_curve(y, y_conf)

plt.figure()
plt.title('Precision-Recall Curve')
plt.plot([0, 1], [0, 0], color='navy', lw=lw, linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, color='darkorange', label='SVM')
plt.legend(loc='lower right')
plt.xlim([-0.1, 1])
plt.ylim([-0.1, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()

