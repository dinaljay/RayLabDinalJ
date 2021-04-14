import os.path
import pandas as pd
import numpy as np

# OUTDATED - NOT IN USE

dhi_features = ["dti_adc_map", "dti_axial_map", "dti_fa_map", "fiber1_axial_map", "fiber1_fa_map",
    "fiber1_radial_map", "fiber_fraction_map", "hindered_fraction_map", "restricted_fraction_map",
                "water_fraction_map", "axon_volume", "inflammation_volume"]

file1 = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/Rachel/CM/csm_dti_adc_map_data.csv"
csm_dti_adc = np.genfromtxt(file1,delimiter=",",skip_header=1)

file2 = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/Rachel/Control/control_dti_adc_map_data.csv"
control_dti_adc = np.genfromtxt(file2,delimiter=",",skip_header=1)

all_dti_adc = np.concatenate((control_dti_adc,csm_dti_adc),axis=0)

print(csm_dti_adc.shape)

print(control_dti_adc.shape)

print(len(all_dti_adc))

#for x in range(len(dhi_features)):
    #print(dhi_features[x])
