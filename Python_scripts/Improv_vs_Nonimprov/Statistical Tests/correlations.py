import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import pearsonr

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

improv_features = ['change_ndi', 'change_dash', 'change_mjoa', 'change_MCS', 'change_PCS']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data_raw = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features + clinical_features
all_data = all_data_raw[all_features]

r_final = []
p_val_final = []

for n in range(len(improv_features)):

    # Set data to variables
    X = all_data.drop(['dti_adc_map', 'dti_axial_map', 'dti_fa_map', 'dti_radial_map'], axis=1)
    y = all_data_raw[improv_features[n]]
    col_names = list(X.columns)

    r = []
    p_val = []
    test = range(len(X.columns))

    for i in range(len(X.columns)):

        #Pearsons_correlation
        col = col_names[i]
        col_data = X[col]
        coeff, p = pearsonr(col_data, y)

        r.append(coeff)
        p_val.append(p)

    r = [round(num, 3) for num in r]
    p_val = [round(val, 3) for val in p_val]

    print(improv_features[n])
    #print('Correlation coefficients:', r)
    #print('p_values:', p_val)
    print("\n")

    r = np.asarray(r)
    p_val = np.asarray(p_val)

    r_final.append(r)
    p_val_final.append(p_val)

#print(r_final)

r_df = pd.DataFrame(data = r_final)
r_df.columns = list(X.columns)
r_df.index = improv_features

p_val_df = pd.DataFrame(data=p_val_final)
p_val_df.columns = list(X.columns)
p_val_df.index = improv_features

print("Done")