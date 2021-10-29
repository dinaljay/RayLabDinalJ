import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy.stats as stats

## Initialize features


radiographic_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "dti_radial_map", "fiber1_axial_map", "fiber1_fa_map",
                         "fiber1_radial_map", "fiber_fraction_map", "hindered_adc_map", "hindered_fraction_map",
                         "iso_adc_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map",
                         "water_fraction_map", "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map",
                         "fiber1_intra_axial_map", "fiber1_intra_fraction_map", "fiber1_intra_radial_map"]

clinical_features = ["babinski_test", "hoffman_test", "avg_right_result", "avg_left_result", "ndi_total", "mdi_total", "dash_total",
                     "PCS", "MCS", "mjoa_total"]

improv_features = ['ndi_improve', 'dash_improve', 'mjoa_improve', 'MCS_improve', 'PCS_improve']

#improv_features = ['ndi_improve', 'MCS_improve']

## Load Data

url = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/dbsi_clinical_radiographic_data.csv'
all_data = pd.read_csv(url, header=0)

# Filter Data
all_features = radiographic_features + clinical_features

#ANOVA test
rows, cols = (len(all_features), len(improv_features))
data_f = [[0]*cols for _ in range(rows)]
data_p = [[0]*cols for _ in range(rows)]


for n in range(len(improv_features)):
    print(improv_features[n])

    for i in range(len(all_features)):
        print(all_features[i])

        is_0 = all_data.loc[all_data[improv_features[n]]==0]
        is_1 = all_data.loc[all_data[improv_features[n]]==1]

        y = is_0[all_features[i]]
        X = is_1[all_features[i]]

        #ANOVA test
        f_value, p_value = stats.f_oneway(X, y)

        #Collect p value and F values
        data_f[i][n] = f_value
        data_p[i][n] = p_value

    print("\n")


row_out = pd.DataFrame(data=all_features, columns=['Feature'])
inter_f = pd.DataFrame(data=data_f, columns=improv_features)
inter_p = pd.DataFrame(data=data_p, columns=improv_features)

#Export dataframe

final_p = pd.concat([row_out, inter_p], axis=1)
final_p.to_csv(r'/home/functionalspinelab/Desktop/Dinal/DBSI_data/anova_p_data.csv', index=False, header=True)

final_f = pd.concat([row_out, inter_f], axis=1)
final_f.to_csv(r'/home/functionalspinelab/Desktop/Dinal/DBSI_data/anova_f_data.csv', index=False, header=True)

#FDR correction

import statsmodels.stats.multitest as fdr
p_corr = [[0]*cols for _ in range(rows)]

for j in range(len(improv_features)):
    temp = final_p[improv_features[j]].to_list()
    h, temp2 = fdr.fdrcorrection(temp, alpha=0.05, method='indep')
    for k in range(len(temp2)):
        p_corr[k][j] = temp2[k]


inter_pcorr = pd.DataFrame(data=p_corr, columns=improv_features)
final_p_corr = pd.concat([row_out, inter_pcorr], axis=1)
final_p_corr.to_csv(r'/home/functionalspinelab/Desktop/Dinal/DBSI_data/anova_pcorr_data.csv', index=False, header=True)

