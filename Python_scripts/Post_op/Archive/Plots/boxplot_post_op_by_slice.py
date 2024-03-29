
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url1 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_1_data.csv'
slice1_data = pd.read_csv(url1)
slice1_data = slice1_data[slice1_data['operation']== 'Post_op']

url2 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_2_data.csv'
slice2_data = pd.read_csv(url2)
slice2_data = slice2_data[slice2_data['operation']== 'Post_op']

url3 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_3_data.csv'
slice3_data = pd.read_csv(url3)
slice3_data = slice3_data[slice3_data['operation']== 'Post_op']

url4 = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_4_data.csv'
slice4_data = pd.read_csv(url4)
slice4_data = slice4_data[slice4_data['operation']== 'Post_op']

dhi_features = ["DTI ADC", "DTI Axial", "DTI FA", "DTI Radial", "Fiber Axial", "Fiber FA ",
    "Fiber Radial", "Fiber Fraction", "Hindered Fraction", "Restricted Fraction", "Water Fraction", "Axon Volume", "Inflammation Volume"]

for i in range(len(dhi_features)):
    filter_data1 = slice1_data[slice1_data['feature_col'] == dhi_features[i]]
    filter_data2 = slice2_data[slice2_data['feature_col'] == dhi_features[i]]
    filter_data3 = slice3_data[slice3_data['feature_col'] == dhi_features[i]]
    filter_data4 = slice4_data[slice4_data['feature_col'] == dhi_features[i]]

    # create grouped boxplot
    plt.figure()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey= True,figsize=(12, 16))
    fig.suptitle(dhi_features[i])

    #slice 1
    sns.boxplot(ax = axes[0,0], x = filter_data1['group'],
                     y = filter_data1['data'], showfliers=False)
    axes[0,0].set_title('Slice 1 - C6')
    axes[0,0].set_ylabel("")
    axes[0,0].set_xlabel("")

    #slice 2
    sns.boxplot(ax = axes[0,1], x = filter_data2['group'],
                     y = filter_data2['data'], showfliers=False)
    axes[0,1].set_title('Slice 2 - C5')
    axes[0,1].set_ylabel("")
    axes[0,1].set_xlabel("")

    #slice 3
    sns.boxplot(ax = axes[1,0], x = filter_data3['group'],
                     y = filter_data3['data'], showfliers=False)
    axes[1,0].set_title('Slice 3 - C4')
    axes[1,0].set_ylabel("")
    axes[1,0].set_xlabel("")

    #slice 4
    sns.boxplot(ax = axes[1,1], x = filter_data4['group'],
                     y = filter_data4['data'], showfliers=False)
    axes[1,1].set_title('Slice 4 - C3')
    axes[1,1].set_ylabel("")
    axes[1,1].set_xlabel("")
    plt.show()


