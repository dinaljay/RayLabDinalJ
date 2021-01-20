% This script calculates the axon and inflammation volume for each slice of
% each patient

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';

out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control';
out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM';
out_dir_mod_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM';

%% Declare necessary variables

% controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];
controls = [4];

mild_cm_subjects = [2,3,5,15,16,18,19,23,28,29,36,40];

moderate_cm_subjects = [9,12,20,27];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);

voxel_volume = 0.35*0.35*7.5;

voxel_surface_area = 0.35*0.35;

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load suppporting mat files

% Axon volume is the product of ROI volume and fiber fraction

% Inflammation volume is the product of inflammation (restricted+hindered
% fraction) and ROI volume

%Volume files
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control/control_volume_data.mat');
control_volumes = cell2mat(control_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM/mild_csm_volume_data.mat');
mild_csm_volumes = cell2mat(mild_csm_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM/mod_csm_volume_data.mat');
mod_csm_volumes = cell2mat(mod_csm_volumes);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control/control_fiber_fraction_map_data.mat');
control_fiber_fraction = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
mild_csm_fiber_fraction = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
mod_csm_fiber_fraction = cell2mat(data_mod_csm);

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control/control_hindered_fraction_map_data.mat');
control_hindered = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
mild_csm_hindered = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
mod_csm_hindered = cell2mat(data_mod_csm);

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Control/control_restricted_fraction_map_data.mat');
control_restricted = cell2mat(data_control);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
mild_csm_restricted = cell2mat(data_mild_csm);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Post_op/ROI/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
mod_csm_restricted = cell2mat(data_mod_csm);

%% Volume Calculations

%Control volume calculations

control_inflammation = control_hindered + control_restricted;

for i = 1:numel(controls)
    control_inflammation_volume{i,1} = control_volumes(i,1)*control_inflammation(i,1);
    control_axon_volume{i,1} = control_volumes(i,1)*control_fiber_fraction(i,1);
end

terminal = strcat('control_inflammation_volume','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_inflammation_volume');

terminal = strcat('control_axon_volume','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_axon_volume');


%Mild CSM volume calculations

mild_csm_inflammation = mild_csm_hindered + mild_csm_restricted;


for i = 1:numel(mild_cm_subjects)
    mild_csm_inflammation_volume{i,1} = mild_csm_volumes(i,1)*mild_csm_inflammation(i,1);
    mild_csm_axon_volume{i,1} = mild_csm_volumes(i,1)*mild_csm_fiber_fraction(i,1);
end

terminal = strcat('mild_csm_inflammation_volume','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_inflammation_volume');

terminal = strcat('mild_csm_axon_volume','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_axon_volume');


%Moderate CSM volume calculations

mod_csm_inflammation = mod_csm_hindered + mod_csm_restricted;


for i = 1:numel(moderate_cm_subjects)
    mod_csm_inflammation_volume{i,1} = mod_csm_volumes(i,1)*mod_csm_inflammation(i,1);
    mod_csm_axon_volume{i,1} = mod_csm_volumes(i,1)*mod_csm_fiber_fraction(i,1);
end

terminal = strcat('mod_csm_inflammation_volume','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_inflammation_volume');

terminal = strcat('mod_csm_axon_volume','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_axon_volume');


