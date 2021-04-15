
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/Pycharm_Data_ROI/all_ROI_data.csv'

all_data = pd.read_csv(url)

pre_op_data = all_data[all_data['operation']== 'Pre_op']

dhi_features = ["DTI ADC", "DTI Axial", "DTI FA", "DTI Radial", "Fiber Axial", "Fiber FA ",
    "Fiber Radial", "Fiber Fraction", "Hindered Fraction", "Restricted Fraction", "Water Fraction", "Axon Volume", "Inflammation Volume"]

for i in range(len(dhi_features)):
    filter_data = pre_op_data[pre_op_data['feature_col'] == dhi_features[i]]

    # create grouped boxplot
    plt.figure()
    ax = sns.boxplot(x = filter_data['group'],
                     y = filter_data['data'], showfliers=False)
    ax.set_title(dhi_features[i])
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.show()


