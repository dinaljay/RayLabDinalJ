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


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map",
                         "fiber1_fa_map",
                         "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                         "restricted_adc_map", "restricted_fraction_map", "water_adc_map",
                         "water_fraction_map", "fiber1_extra_axial_map", "fiber1_extra_fraction_map",
                         "fiber1_extra_radial_map",
                         "fiber1_intra_axial_map", "fiber1_intra_fraction_map", "fiber1_intra_radial_map"]

clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total",
                     "dash_total",
                     "PCS", "MCS", "mjoa_total", "Elix_1", "Elix_2", "Elix_3", "Elix_4", "Elix_5", "smoking"]

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features
all_data = all_data_raw[all_features]
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
#print(ev)

plt.scatter(range(1, X_scaled.shape[1] + 1), ev)
plt.plot(range(1, X_scaled.shape[1] + 1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigen Value')
plt.grid()
plt.show()
