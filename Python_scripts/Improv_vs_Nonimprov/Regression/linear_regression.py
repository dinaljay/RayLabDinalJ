import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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

improv_features_lr = ['change_ndi', 'change_dash', 'change_mjoa', 'change_MCS', 'change_PCS']

improv_features_svm = ['ndi_improve', 'dash_improve', 'mjoa_improve', 'MCS_improve', 'PCS_improve']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features + clinical_features
all_data = all_data_raw[all_features]

for n in range(len(improv_features_svm)):

    # Set data to variables
    X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
    y = all_data_raw[improv_features_svm[n]]

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
    LR = LinearRegression()
    selector = RFE(estimator=LR, step=1, n_features_to_select=1)
    final = selector.fit(X_scaled, y)

    features = list(X.columns)

    d = {'Feature': features, 'Ranking': selector.ranking_}
    rankings = pd.DataFrame(data=d)
    rankings = rankings.sort_values(by=['Ranking'])

    #Create list of rfe_features
    rfe_features = rankings["Feature"].tolist()
    rfe_features = rfe_features[0:10]

    print(improv_features_svm[n])
    print(rfe_features)

    #Set data to variables
    X = all_data[rfe_features]
    y = all_data_raw[improv_features_lr[n]]
    del rfe_features
    # Scale data
    X_scaled = preprocessing.scale(X)

    #Print model variables
    func = LinearRegression()
    func.fit(X_scaled, y)
    print('Regression coefficient:', func.coef_)

    #Implement leave one out cross validation
    y_pred = []
    y_conf = []

    for i in range(len(X_scaled)):
        # Splitting Data for tuning hyerparameters
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = y.drop([i], axis=0)

        # Splitting Data for model
        X_train = np.delete(X_scaled, [i], axis=0)
        y_train = y.drop([i], axis=0)

        X_test = X_scaled[i]
        X_test = X_test.reshape(1, -1)
        y_test = y[i]

        # Generating linear regression model
        clf = LinearRegression()

        # Train the model using the training sets
        clf.fit(X_train, y_train)

        # Predict the response for test dataset
        temp = clf.predict(X_test)
        y_pred.append(temp[0])


    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    # Model R2 score
    print("Accuracy:", metrics.r2_score(y, y_pred))

    # Model Mean Squared Error
    print("Mean squared error:", metrics.mean_squared_error(y, y_pred))

    #Model Root Mean Squared Array
    print("Root mean squared error:", np.sqrt(metrics.mean_squared_error(y, y_pred)))

    print("\n")




