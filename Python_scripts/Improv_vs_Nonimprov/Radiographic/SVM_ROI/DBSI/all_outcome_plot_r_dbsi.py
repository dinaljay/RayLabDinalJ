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


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map", "fiber1_fa_map",
                         "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                         "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map",
                         "water_fraction_map", "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map",
                         "fiber1_intra_axial_map", "fiber1_intra_fraction_map", "fiber1_intra_radial_map"]

#improv_features = ['ndi_improve', 'dash_improve', 'mjoa_improve', 'MCS_improve', 'PCS_improve']

improv_features = ['mjoa_improve']


## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features
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

    # Tuning hyperparameters
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    clf = GridSearchCV(SVC(), tuned_parameters, scoring='accuracy')
    clf.fit(X_scaled, y)
    params = clf.best_params_
    cost = params['C']

    #RFE
    svc = SVC(kernel="linear", C=cost)
    selector = RFE(estimator=svc, step=1, n_features_to_select=1)
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

    precision[n], recall[n], _ = metrics.precision_recall_curve(y.ravel(), y_conf.ravel())
    prc_auc[n] = metrics.auc(recall[n], precision[n])

#colors = cycle(['darkorange', 'red', 'green', 'navy', 'purple'])

colors = cycle(['green'])


#sys.exit()
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




