
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys

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

## Load Pre-op Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_mild_CSM_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop(['Patient_ID', 'Group', 'Group_ID', 'dti_adc', 'dti_axial', 'dti_fa', 'dti_radial'], axis=1)
#X = all_data[['fiber_fraction', 'fiber_fa', 'fiber_radial']]
y = all_data['Group_ID']

# Scale data

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

# Generating SVM model
clf = SVC(C=cost, kernel="linear")

#Train the model using the training sets
clf.fit(X_scaled, y)

## Load Post-op Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/Pycharm_Data_ROI/DBSI_CSV_Data/Post_op/all_patients_all_features_mild_CSM_data.csv'

all_data = pd.read_csv(url, header=0)

X = all_data.drop(['Patient_ID', 'Group', 'Group_ID', 'dti_adc', 'dti_axial', 'dti_fa', 'dti_radial'], axis=1)
#X = all_data[['fiber_fraction', 'fiber_fa', 'fiber_radial']]
y = all_data['Group_ID']

# Scale data

from sklearn import preprocessing

X_scaled = preprocessing.scale(X)
X_scaled = np.asarray(X_scaled)
y_pred = []
y_conf = []

for i in range(len(X_scaled)):

    #Predict the response for test dataset
    hold = X_scaled[i]
    hold = hold.reshape(1, -1)
    temp = clf.predict(hold)
    y_pred.append(temp[0])

    #Get confidecne scores
    temp = clf.decision_function(hold)
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

#Save dataframe of y and y_pred as csv file
all_data = pd.read_csv(url)
out_folder = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/Post_op_predictions/hc_vs_mild_csm.csv'
data = all_data.iloc[:, :2]
temp1 = y.reshape(len(y), 1)
temp2 = y_pred.reshape(len(y_pred), 1)
y_df = pd.DataFrame(temp1, columns=['Group ID'])
y_pred_df = pd.DataFrame(temp2, columns=['Pred Group ID'])
out_df = pd.concat([data, y_df, y_pred_df], axis=1)
out_df.to_csv(out_folder, index=False, header=True)


sys.exit()
#Plot ROC curve

lw=2
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='SVM (area = %0.2f)' %roc_auc)
plt.plot([0, 1], [0, 1],color='navy', lw=lw, linestyle='--', label='No Skill')
plt.legend(loc='lower right')
plt.xlim([-0.05, 1])
plt.ylim([-0.05, 1.05])
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
plt.xlim([-0.05, 1])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()

