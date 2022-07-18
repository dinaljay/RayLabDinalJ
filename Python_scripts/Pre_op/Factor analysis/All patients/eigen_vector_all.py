import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import metrics
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
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

## Bartlett's test of sphericity

# chi_square_value, p_value = calculate_bartlett_sphericity(X_scaled)
# print(chi_square_value, p_value)

## KMO test

# kmo_all, kmo_model = calculate_kmo(X_scaled)
# print(kmo_model)

## Create factor analysis object and perform factor analysis
# fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=X.shape[1])
fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=10)
fa.fit(X_scaled)

## Check eigen values
ev, _ = fa.get_eigenvalues()
print(ev)

plt.scatter(range(1, X_scaled.shape[1] + 1), ev)
plt.plot(range(1, X_scaled.shape[1] + 1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()
plt.show()
