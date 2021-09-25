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

from sklearn.metrics import classification_report, make_scorer, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import random

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=109, shuffle=True, stratify=y) # 70% training and 30% test and validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=109, shuffle=True, stratify=y_temp) # 66.66% validation and 30% test

#Grid search parameters

dense_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dropout = [0.001, 0.1, 0.2, 0.3, 0.4]
optimizer = ['RMSprop', 'sgd', 'adam']
batch_size = [100, 150, 200]
epochs = [100, 150, 200, 250]

param_grid = dict(dense_layers=dense_layers, dropout=dropout, optimizer=optimizer,
                    batch_size=batch_size, epochs=epochs)
OUTPUT_CLASSES = 1

def model_fn(dense_layers=1, dropout=0.1, optimizer='adam'):
    """Create a Keras model with the given hyperparameters.
    Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
    Returns:
      A compiled Keras model.
    """

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(18, input_dim=18, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))

    # Add fully connected layers.

    for _ in range(dense_layers):
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout))

    # Add the final output layer.
    model.add(tf.keras.layers.Dense(OUTPUT_CLASSES, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


# create model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=model_fn, verbose=0)
kfold = KFold(n_splits=3, random_state=42)

# define the grid search parameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold, verbose=3)
grid_result = grid.fit(X_val, y_val)

# Print best results
print('\n')
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

sys.exit()

# Define cross validation
kfold = KFold(n_splits=5, random_state=42)

# AUC and accuracy as score
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}

# Define grid search
grid = GridSearchCV(
  estimator=KerasClassifier(build_fn=model_fn(), verbose=0),
  param_grid=search_space,
  cv=kfold,
  scoring=scoring,
  refit='AUC',
  verbose=0,
  n_jobs=-1
)
# Fit grid search
model = grid.fit(X_val, y_val)

predict = model.predict(X_test)
print('Best AUC Score: {}'.format(model.best_score_))
print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
print(confusion_matrix(y_test,predict))

#Print best parameters
print('\n')
print(model.best_params_)