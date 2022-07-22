% This script calculates the axon and inflammation volume for each slice of
% each patient

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved';
out_dir_non_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved';

%% Declare necessary variables

improv_subjects = [2,4,5,9,10,13,14,20,19,21,22,26,30,34,36,40,41,42,43,44,46,49]; %mJOA
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46]; %NASS

non_improv_subjects = [3,6,11,12,15,16,18,23,24,25,27,28,29,31,32,37,38,45,48,50];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF
% non_improv_subjects = [12,18,20,24,40]; %NASS6 PF

slices = (1:1:4);
% slices = [4];

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_dirz_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_axial_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...    
    "fiber1_fa_map";"fiber1_fiber_fraction_map";"fiber1_radial_map";"fiber1_rgba_map";"fiber1_rgba_map_itk";...
    "fiber2_axial_map";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_fa_map";...
    "fiber2_fiber_fraction_map";"fiber2_radial_map";"fiber_fraction_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

%% Load suppporting mat files

% Axon volume is the product of ROI volume and fiber fraction

% Inflammation volume is the product of inflammation (restricted+hindered
% fraction) and ROI volume

%Volume files
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_volume_data.mat');
improv_volumes = cell2mat(improv_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_volume_data.mat');
non_improv_volumes = cell2mat(non_improv_volumes);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber_fraction_map_data.mat');
improv_fiber_fraction = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber_fraction_map_data.mat');
non_improv_fiber_fraction = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_hindered_fraction_map_data.mat');
improv_hindered = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_hindered_fraction_map_data.mat');
non_improv_hindered = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_restricted_fraction_map_data.mat');
improv_restricted = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_restricted_fraction_map_data.mat');
non_improv_restricted = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

%% Volume Calculations

%Mild CSM volume calculations

improv_inflammation = improv_hindered + improv_restricted;

for i = 1:numel(improv_subjects)
    improv_inflammation_volume{i,1} = improv_volumes(i,1)*improv_inflammation(i,1);
    improv_axon_volume{i,1} = improv_volumes(i,1)*improv_fiber_fraction(i,1);
end

terminal = strcat('improv_inflammation_volume','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_inflammation_volume');

terminal = strcat('improv_axon_volume','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_axon_volume');


%Moderate CSM volume calculations

non_improv_inflammation = non_improv_hindered + non_improv_restricted;


for i = 1:numel(non_improv_subjects)
    non_improv_inflammation_volume{i,1} = non_improv_volumes(i,1)*non_improv_inflammation(i,1);
    non_improv_axon_volume{i,1} = non_improv_volumes(i,1)*non_improv_fiber_fraction(i,1);
end

terminal = strcat('non_improv_inflammation_volume','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_inflammation_volume');

terminal = strcat('non_improv_axon_volume','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_axon_volume');

