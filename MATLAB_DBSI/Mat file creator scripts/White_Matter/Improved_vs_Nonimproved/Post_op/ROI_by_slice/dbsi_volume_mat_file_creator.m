% This script calculates the axon and inflammation volume for each slice of
% each patient

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Improved';
out_dir_non_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Nonimproved';

%% Declare necessary variables

improv_subjects = [2,5,9,24,26,30,36,40,41,46]; %mJOA
% % improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF

non_improv_subjects = [3,12,15,18,20,21,23,27,28,29,45,];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"dti_radial_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load suppporting mat files

% Axon volume is the product of ROI volume and fiber fraction

% Inflammation volume is the product of inflammation (restricted+hindered
% fraction) and ROI volume

%Volume files
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Improved/improv_volume_data.mat');
improv_volumes = cell2mat(improv_volumes);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Nonimproved/non_improv_volume_data.mat');
non_improv_volumes = cell2mat(non_improv_volumes);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Improved/improv_fiber_fraction_map_data.mat');
improv_fiber_fraction = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Nonimproved/non_improv_fiber_fraction_map_data.mat');
non_improv_fiber_fraction = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Improved/improv_hindered_fraction_map_data.mat');
improv_hindered = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Nonimproved/non_improv_hindered_fraction_map_data.mat');
non_improv_hindered = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Improved/improv_restricted_fraction_map_data.mat');
improv_restricted = cell2mat(data_improv);
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Post_op/ROI_by_slice/Nonimproved/non_improv_restricted_fraction_map_data.mat');
non_improv_restricted = cell2mat(data_non_improv);
clear data_control; clear data_improv; clear data_non_improv;

%% Volume Calculations

%Improved volume calculations

improv_inflammation = improv_hindered + improv_restricted;

for i = 1:numel(improv_subjects)
    for j = 1:numel(slices)
        improv_inflammation_volume{i,j} = improv_volumes(i,j)*improv_inflammation(i,j);
        improv_axon_volume{i,j} = improv_volumes(i,j)*improv_fiber_fraction(i,j);
    end
end

terminal = strcat('improv_inflammation_volume','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_inflammation_volume');

terminal = strcat('improv_axon_volume','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_axon_volume');


%Non-improved volume calculations

non_improv_inflammation = non_improv_hindered + non_improv_restricted;


for i = 1:numel(non_improv_subjects)
    for j = 1:numel(slices)
        non_improv_inflammation_volume{i,j} = non_improv_volumes(i,j)*non_improv_inflammation(i,j);
        non_improv_axon_volume{i,j} = non_improv_volumes(i,j)*non_improv_fiber_fraction(i,j);
    end
end

terminal = strcat('non_improv_inflammation_volume','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_inflammation_volume');

terminal = strcat('non_improv_axon_volume','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_axon_volume');


