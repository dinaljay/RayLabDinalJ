import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import metrics
from factor_analyzer.factor_analyzer import FactorAnalyzer

## Initialize features

filter_dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map", "fiber1_fa_map",
                       "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                       "restricted_adc_map", "restricted_fraction_map", "water_adc_map", "water_fraction_map"]

filter_dbsi_ia_features = ["fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",
                           "fiber1_intra_radial_map"]

## Load Data

url_dhi = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_by_CSM_group_data.csv'
all_data_dhi = pd.read_csv(url_dhi, header=0)

url_dbsi_ia = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op/all_patients_all_features_by_CSM_group_data.csv'
all_data_dbsi_ia = pd.read_csv(url_dbsi_ia, header=0)

# Filter Data
filter_dhi = all_data_dhi[filter_dhi_features]
filter_dbsi_ia = all_data_dbsi_ia[filter_dbsi_ia_features]

all_data = pd.concat([filter_dhi, filter_dbsi_ia], axis=1)
all_data = all_data.dropna()

X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)

# Scale data
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

## Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation='varimax', n_factors=5)
fa.fit(X_scaled)

## Print loadings
# print(fa.loadings_)
print(pd.DataFrame(fa.loadings_, index=X.columns))
print("\n")

## Rotation matrix
#print(fa.rotation_matrix_)
#print("\n")

## Variance explained by each factor
print(pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var']))
print("\n")

## Communalities
print(pd.DataFrame(fa.get_communalities(), index=X.columns, columns=['Communalities']))
