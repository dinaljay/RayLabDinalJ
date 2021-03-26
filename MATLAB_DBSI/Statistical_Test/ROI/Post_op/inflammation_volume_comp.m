%This script does a non-parametric comparison of inflammation volume
% coefficients for each slice between CSM and control  patients

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB Scripts'));

%% Import data

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM/mild_csm_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM/mod_csm_inflammation_volume_data.mat');

%% U test implementation

mild_cm = cell2mat(data_mild_csm(:,1));
mod_cm = cell2mat(data_mod_csm(:,1));
control = cell2mat(data_control(:,1));

[p_mild,~,stats_array] = ranksum(mild_cm,control,'method','approximate');
zval_mild = stats_array.zval;

[p_mod,~,stats_array] = ranksum(mod_cm,control,'method','approximate');
zval_mod = stats_array.zval;

[p_cm,~,stats_array] = ranksum(mild_cm,mod_cm,'method','approximate');
zval_cm = stats_array.zval;