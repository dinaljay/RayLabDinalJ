% This script does a rank sum test to see if there is a significant
% difference in age between controls and CSM patients

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/Clinical_data/csm_ages.csv';
control_path = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/Clinical_data/control_ages.csv';

%% Load data

csm_ages = importdata(csm_path);
control_ages = importdata(control_path);

csm = csm_ages.data;
control = control_ages;

%% ranksum test

[p_age,~,stats_array_age] = ranksum(csm,control,'method','approximate');
zval_age = stats_array_age.zval;

