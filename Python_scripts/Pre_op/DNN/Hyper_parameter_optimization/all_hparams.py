import os.path as path
import matplotlib.pyplot as plt
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

# Load Data

url_dhi = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_Voxel/Pre_op/all_patients_all_features_data.csv'
all_data_dhi = pd.read_csv(url_dhi, header=0)
all_data_dhi[all_data_dhi['Group_ID'] == 2] = 1

url_dbsi_ia = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_Voxel/Pre_op/all_patients_all_features_data.csv'
all_data_dbsi_ia = pd.read_csv(url_dbsi_ia, header=0)
all_data_dbsi_ia[all_data_dbsi_ia['Group_ID'] == 2] = 1

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

X_scaled = preprocessing.scale(X)

#Import DNN databases

import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=109, shuffle=True) # 70% training and 30% test and validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=109, shuffle=True) # 66.66% validation and 30% test

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete[16, 32])
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.001, 0.3))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RSMprop']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.tanh),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
  ])
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  model.fit(X_val, y_val, epochs=50, batch_size=150) # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(X_val, y_val)
  return accuracy

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1


#%tensorboard --logdir logs/hparam_tuning
