
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
from sklearn.decomposition import PCA

scaler = preprocessing.StandardScaler()
pca = PCA(n_components=2)
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
wcss = []

for i in range(1, 11):

    #Initialize kmeans
    kmeans = KMeans(
        init="random",
        n_clusters=i,
        n_init=10,
        max_iter=300,
        random_state=42)

    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Elbow method")
plt.xlabel('Number of clusters')
plt.ylabel("WCSS")
plt.show()