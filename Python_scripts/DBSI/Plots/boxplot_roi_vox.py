
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

## Load Data

url = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI_Voxel/Pre_op/all_patients_all_features_by_CSM_group_data.csv'

all_data = pd.read_csv(url, header=0)

dhi_features = ["dti_adc", "dti_axial", "dti_fa", "dti_radial", "fiber_axial", "fiber_fa",
    "fiber_radial", "fiber_fraction", "hindered_fraction", "restricted_fraction",
                "water_fraction"]

#groups = ["Control", "MildCSM", "Moderate CSM"]
#Boxplot

fig, axes = plt.subplots(6, 2, figsize=(18, 10), sharex=True)
#axes = axes.ravel()
sns.boxplot(ax=axes[0, 0], x=all_data["Group"], y=all_data[dhi_features[0]], width=0.3)
axes[0, 0].set_title(dhi_features[0])

sns.boxplot(ax=axes[0, 1], x=all_data["Group"], y=all_data[dhi_features[1]], width=0.3)
axes[0, 1].set_title(dhi_features[1])

sns.boxplot(ax=axes[1, 0], x=all_data["Group"], y=all_data[dhi_features[2]], width=0.3)
axes[1, 0].set_title(dhi_features[2])

sns.boxplot(ax=axes[1, 1], x=all_data["Group"], y=all_data[dhi_features[3]], width=0.3)
axes[1, 1].set_title(dhi_features[3])

sns.boxplot(ax=axes[2, 0], x=all_data["Group"], y=all_data[dhi_features[4]], width=0.3)
axes[2, 0].set_title(dhi_features[4])

sns.boxplot(ax=axes[2, 1], x=all_data["Group"], y=all_data[dhi_features[5]], width=0.3)
axes[2, 1].set_title(dhi_features[5])

sns.boxplot(ax=axes[3, 0], x=all_data["Group"], y=all_data[dhi_features[6]], width=0.3)
axes[3, 0].set_title(dhi_features[6])

sns.boxplot(ax=axes[3, 1], x=all_data["Group"], y=all_data[dhi_features[7]], width=0.3)
axes[3, 1].set_title(dhi_features[7])

sns.boxplot(ax=axes[4, 0], x=all_data["Group"], y=all_data[dhi_features[8]], width=0.3)
axes[4, 0].set_title(dhi_features[8])

sns.boxplot(ax=axes[4, 1], x=all_data["Group"], y=all_data[dhi_features[9]], width=0.3)
axes[4, 1].set_title(dhi_features[9])

sns.boxplot(ax=axes[5, 0], x=all_data["Group"], y=all_data[dhi_features[10]], width=0.3)
axes[5, 0].set_title(dhi_features[10])

plt.show()
sys.exit()

for i in range(len(dhi_features)):
    sns.boxplot(x=all_data["Group"], y=all_data[dhi_features[i]], width=0.3, palette='Set3')
    axes[i].set_title(dhi_features[i])

plt.show()

sys.exit()
fig = plt.figure(figsize=(18, 10))

for i in range(len(dhi_features)):
    pos=i+1
    plt.subplot(5, 2, pos, sharey=True)
    plt.title(dhi_features[i])
    sns.boxplot(x=all_data["Group"], y=all_data[dhi_features[i]], width=0.3, palette='Set3')

plt.show()

sys.exit()

#Additional testing
for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
