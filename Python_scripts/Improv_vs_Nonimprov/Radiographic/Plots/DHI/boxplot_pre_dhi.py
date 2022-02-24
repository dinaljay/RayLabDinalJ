
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_improv_vs_nonimprov/all_ROI_data.csv'

all_data = pd.read_csv(url)

dhi_features = ["b0_map", "dti_adc_map", "dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_dirz_map", "dti_fa_map",
                "dti_g_map", "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_axial_map", "fiber1_dirx_map",
                "fiber1_diry_map", "fiber1_dirz_map", "fiber1_fa_map", "fiber1_fiber_fraction_map", "fiber1_radial_map", "fiber1_rgba_map",
                "fiber1_rgba_map_itk", "fiber2_axial_map", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_fa_map",
                "fiber2_fiber_fraction_map", "fiber2_radial_map", "fiber_fraction_map", "fraction_rgba_map", "hindered_adc_map",
                "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map",
                "water_adc_map", "water_fraction_map"]

#dhi_features = ["DTI ADC", "DTI Axial", "DTI FA", "DTI Radial", "Fiber Axial", "Fiber FA ",
#    "Fiber Radial", "Fiber Fraction", "Hindered Fraction", "Restricted Fraction", "Water Fraction", "Axon Volume", "Inflammation Volume"]

for i in range(len(dhi_features)):
    filter_data = all_data[all_data['feature_col'] == dhi_features[i]]

    # create grouped boxplot
    plt.figure(figsize=(8,7.5))
    ax = sns.boxplot(x = filter_data['group'],
                     y = filter_data['data'], showfliers=False)
    ax.set_title(dhi_features[i])
    ax.set_ylabel("")
    ax.set_xlabel("")

    lines = []
    labels = []

    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

    plt.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.25,-0.1,0.5,0.5), ncol=2)
    plt.show()


