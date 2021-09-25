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
input_size = X_scaled.shape[1]
#Import DNN databases

import numpy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.callbacks import TensorBoard
import time

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=109, shuffle=True, stratify=y) # 70% training and 30% test and validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=109, shuffle=True, stratify=y_temp) # 66.66% validation and 30% test

#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([12]))
#HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.Discrete([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(['RMSprop', 'sgd', 'adam']))
#HP_ACTIVATION = hp.HParam("activation", hp.Discrete(['softmax',  'relu', 'tanh', 'sigmoid', 'linear']))
#HP_BATCH = hp.HParam("batch_size", hp.Discrete([100, 125, 150, 175, 200]))
#HP_EPOCH = hp.HParam("epochs", hp.Discrete([100, 150, 200]))


METRICS = [
    hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "batch_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "batch_loss",
        group="train",
        display_name="loss (train)",
    ),
]

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('/media/functionalspinelab/RAID/Data/Dinal/DNN_logs/binary/v1/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

def train_test_model(hparams):
    # define the keras model

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # Input layer
    model.add(tf.keras.layers.Dense(18, input_dim=input_size, activation='relu'))
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))

    # Layer 1
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))

    # Layer 2
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT]))

    # Add final output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    NAME = "DBSI-DNN-{}".format(int(time.time()))
    tensorboard = TensorBoard(log_dir="/media/functionalspinelab/RAID/Data/Dinal/DNN_logs/binary/suyash/{}".format(NAME))

    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(X_val, y_val, epochs=200, batch_size=100, callbacks=[tensorboard])  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0

for dropout_rate in HP_DROPOUT.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
        HP_DROPOUT: dropout_rate,
        HP_OPTIMIZER: optimizer,
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('/media/functionalspinelab/RAID/Data/Dinal/DNN_logs/binary/suyash/hparam_tuning/' + run_name, hparams)
        session_num += 1