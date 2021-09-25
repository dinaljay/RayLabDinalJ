
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url1 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_1_data.csv'
slice1_data = pd.read_csv(url1)

url2 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_2_data.csv'
slice2_data = pd.read_csv(url2)

url3 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_3_data.csv'
slice3_data = pd.read_csv(url3)

url4 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_4_data.csv'
slice4_data = pd.read_csv(url4)

dhi_features = ["b0_map", "dti_adc_map", "dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_fa_map", "dti_g_map",\
    "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_dirx_map", "fiber1_diry_map", "fiber1_dirz_map",\
    "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",\
    "fiber1_intra_radial_map", "fiber1_rgba_map_itk", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_extra_axial_map",\
    "fiber2_extra_fraction_map", "fiber2_extra_radial_map", "fiber2_intra_axial_map", "fiber2_intra_fraction_map", "fiber2_intra_radial_map",\
    "fraction_rgba_map", "hindered_adc_map", "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map",\
    "water_adc_map", "water_fraction_map"]


for i in range(len(dhi_features)):
    filter_data1 = slice1_data[slice1_data['MRI_Feature'] == dhi_features[i]]
    filter_data2 = slice2_data[slice2_data['MRI_Feature'] == dhi_features[i]]
    filter_data3 = slice3_data[slice3_data['MRI_Feature'] == dhi_features[i]]
    filter_data4 = slice4_data[slice4_data['MRI_Feature'] == dhi_features[i]]

    # create grouped boxplot
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(12, 25))
    fig.suptitle(dhi_features[i])

    #slice 1
    sns.boxplot(ax=axes[0,0], x=filter_data1['Visit'],
                     y=filter_data1['Data_Value'],
                     hue=filter_data1['Patient_Group'], showfliers=False)
    axes[0,0].set_title('Slice 1 - C6')
    axes[0,0].set_ylabel("")
    axes[0,0].set_xlabel("")
    axes[0,0].get_legend().remove()

    #sys.exit()

    #slice 2
    sns.boxplot(ax = axes[0,1], x = filter_data2['Visit'],
                     y = filter_data2['Data_Value'],
                     hue = filter_data2['Patient_Group'], showfliers=False)
    axes[0,1].set_title('Slice 2 - C5')
    axes[0,1].set_ylabel("")
    axes[0,1].set_xlabel("")
    axes[0,1].get_legend().remove()

    #slice 3
    sns.boxplot(ax = axes[1,0], x = filter_data3['Visit'],
                     y = filter_data3['Data_Value'],
                     hue = filter_data3['Patient_Group'], showfliers=False)
    axes[1,0].set_title('Slice 3 - C4')
    axes[1,0].set_ylabel("")
    axes[1,0].set_xlabel("")
    axes[1,0].get_legend().remove()

    #slice 4
    sns.boxplot(ax = axes[1,1], x = filter_data4['Visit'],
                     y = filter_data4['Data_Value'],
                     hue = filter_data4['Patient_Group'], showfliers=False)
    axes[1,1].set_title('Slice 4 - C3')
    axes[1,1].set_ylabel("")
    axes[1,1].set_xlabel("")
    axes[1,1].get_legend().remove()

    lines = []
    labels = []

    axLine, axLabel = axes[0,0].get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

    fig.legend(lines, labels, loc='lower center', ncol=3)
    plt.show()