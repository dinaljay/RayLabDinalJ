% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

improv_subjects = [2,5,9,24,26,30,36,40,41,46]; %mJOA
% % improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF

non_improv_subjects = [3,12,15,18,20,21,23,27,28,29,45,];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF
slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"dti_radial_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map";...
    "axon_volume";"inflammation_volume"];

%% Create variable stores
%B0 Map

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_b0_map_data.mat');
b0 = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_radial_map_data.mat');
dti_radial = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_improv);cell2mat(data_non_improv)];
clear data_improv; clear data_non_improv;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_axon_volume_data.mat');
axon_volume = [cell2mat(improv_axon_volume);cell2mat(non_improv_axon_volume)];
clear improv_axon_volume; clear non_improv_axon_volume;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_inflammation_volume_data.mat');
inflammation_volume = [cell2mat(improv_inflammation_volume);cell2mat(non_improv_inflammation_volume)];
clear improv_inflammation_volume; clear non_improv_inflammation_volume;

%% Create patient IDs
x=1;

for k = 1:numel(improv_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(improv_subjects(k)));
    x=x+1;
end

for k = 1:numel(non_improv_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(non_improv_subjects(k)));
    x=x+1;
end

%% Save csv file

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/Pycharm_Data_improv_vs_nonimprov/Pre_op/ROI';

%Controls vs all CSM

group = categorical([repmat({'Improved'},numel(improv_subjects),1);repmat({'Non-Improved'},numel(non_improv_subjects),1)]);

group_id = categorical([repmat({'0'},numel(improv_subjects),1);repmat({'1'},numel(non_improv_subjects),1)]);

terminal2 = strcat('all_patients_all_features','_data.csv');
table_out=table(patientID, group,group_id,dti_adc,dti_axial,dti_fa,dti_radial,fiber_axial,fiber_fa,...
    fiber_radial,fiber_fraction,hindered_fraction,restricted_fraction,water_fraction,axon_volume,inflammation_volume);

table_out.Properties.VariableNames{1} = 'Patient_ID';
table_out.Properties.VariableNames{2} = 'Group';
table_out.Properties.VariableNames{3} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

