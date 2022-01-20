
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_improv_vs_nonimprov/all_ROI_data.csv'

all_data = pd.read_csv(url)

dhi_features = ["DTI ADC", "DTI Axial", "DTI FA", "DTI Radial", "Fiber Axial", "Fiber FA ",
    "Fiber Radial", "Fiber Fraction", "Hindered Fraction", "Restricted Fraction", "Water Fraction", "Axon Volume", "Inflammation Volume"]

for i in range(len(dhi_features)):
    filter_data = all_data[all_data['feature_col'] == dhi_features[i]]

    # create grouped boxplot
    plt.figure(figsize=(8,7.5))
    ax = sns.boxplot(x = filter_data['operation'],
                     y = filter_data['data'],
                     hue = filter_data['group'], showfliers=False)
    ax.set_title(dhi_features[i])
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.get_legend().remove()

    lines = []
    labels = []

    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

    plt.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.25,-0.1,0.5,0.5), ncol=2)
    plt.show()


