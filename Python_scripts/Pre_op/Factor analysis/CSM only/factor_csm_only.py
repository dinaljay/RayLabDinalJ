import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import metrics
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

## Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation='varimax', n_factors=5)
fa.fit(X_scaled)

## Print loadings
# print(fa.loadings_)
loadings = pd.DataFrame(fa.loadings_, index=X.columns, columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5'])
print(loadings)
print("\n")

## Rotation matrix
#print(fa.rotation_matrix_)
#print("\n")

## Variance explained by each factor
variance = pd.DataFrame(fa.get_factor_variance(), index=['Variance', 'Proportional Var', 'Cumulative Var'], columns=['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5'])
print(variance)
print("\n")

## Communalities
communality = pd.DataFrame(fa.get_communalities(), index=X.columns, columns=['Communalities'])
print((communality))

with pd.ExcelWriter(r'/home/functionalspinelab/Desktop/Factor_analysis/factor_analysis_csm_only.xlsx') as writer:
    loadings.to_excel(writer, sheet_name='Factor Loadings', index=True, header=True)
    variance.to_excel(writer, sheet_name='Variance', index=True, header=True)
    communality.to_excel(writer, sheet_name='Communalities', index=True, header=True)
