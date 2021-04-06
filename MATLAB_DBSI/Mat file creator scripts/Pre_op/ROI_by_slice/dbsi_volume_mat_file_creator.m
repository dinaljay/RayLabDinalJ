% This script calculates the axon and inflammation volume for each slice of
% each patient

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';

out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Control';
out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Mild_CSM';
out_dir_mod_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Moderate_CSM';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];
%CSM_P01 template no good

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);

voxel_volume = 0.35*0.35*7.5;

voxel_surface_area = 0.35*0.35;

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load suppporting mat files

% Axon volume is the product of ROI_by_slice volume and fiber fraction

% Inflammation volume is the product of inflammation (restricted+hindered
% fraction) and ROI_by_slice volume

%Volume files
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Control/control_volume_data.mat');
control_volumes = cell2mat(control_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_volume_data.mat');
mild_csm_volumes = cell2mat(mild_csm_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_volume_data.mat');
mod_csm_volumes = cell2mat(mod_csm_volumes);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Control/control_fiber_fraction_map_data.mat');
control_fiber_fraction = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
mild_csm_fiber_fraction = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
mod_csm_fiber_fraction = cell2mat(data_mod_csm);
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Control/control_hindered_fraction_map_data.mat');
control_hindered = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
mild_csm_hindered = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
mod_csm_hindered = cell2mat(data_mod_csm);
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Control/control_restricted_fraction_map_data.mat');
control_restricted = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
mild_csm_restricted = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
mod_csm_restricted = cell2mat(data_mod_csm);
clear data_control; clear data_mild_csm; clear data_mod_csm;

%% Volume Calculations

%Control volume calculations

control_inflammation = control_hindered + control_restricted;

for i = 1:numel(controls)
    for j = 1:numel(slices)
        control_inflammation_volume{i,j} = control_volumes(i,j)*control_inflammation(i,j);
        control_axon_volume{i,j} = control_volumes(i,j)*control_fiber_fraction(i,j);
    end
end

terminal = strcat('control_inflammation_volume','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_inflammation_volume');

terminal = strcat('control_axon_volume','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_axon_volume');


%Mild CSM volume calculations

mild_csm_inflammation = mild_csm_hindered + mild_csm_restricted;


for i = 1:numel(mild_cm_subjects)
    for j = 1:numel(slices)
        mild_csm_inflammation_volume{i,j} = mild_csm_volumes(i,j)*mild_csm_inflammation(i,j);
        mild_csm_axon_volume{i,j} = mild_csm_volumes(i,j)*mild_csm_fiber_fraction(i,j);
    end
end

terminal = strcat('mild_csm_inflammation_volume','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_inflammation_volume');

terminal = strcat('mild_csm_axon_volume','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_axon_volume');


%Moderate CSM volume calculations

mod_csm_inflammation = mod_csm_hindered + mod_csm_restricted;


for i = 1:numel(moderate_cm_subjects)
    for j = 1:numel(slices)
        mod_csm_inflammation_volume{i,j} = mod_csm_volumes(i,j)*mod_csm_inflammation(i,j);
        mod_csm_axon_volume{i,j} = mod_csm_volumes(i,j)*mod_csm_fiber_fraction(i,j);
    end
end

terminal = strcat('mod_csm_inflammation_volume','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_inflammation_volume');

terminal = strcat('mod_csm_axon_volume','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_axon_volume');


