% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map";...
    "axon_volume";"inflammation_volume"];

%% Create variable stores
%B0 Map

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_b0_map_data.mat');
b0 = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
dti_radial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
% clear data_control; clear data_mild_csm; clear data_mod_csm;

% %Axon volume
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_axon_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_axon_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_axon_volume_map_data.mat');
% axon_volume = [cell2mat(control_axon_volume);cell2mat(mild_csm_axon_volume);cell2mat(mod_csm_axon_volume)];
% 
% %Inflammation volume
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control/control_inflammation_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_inflammation_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_inflammation_volume_map_data.mat');
% inflammation_volume = [cell2mat(control_inflammation_volume);cell2mat(mild_csm_inflammation_volume);cell2mat(mod_csm_inflammation_volume)];

%% Save csv file

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_Voxel/Pre_op';

%Controls vs all CSM

group = categorical([repmat({'Control'},length(data_control),1);repmat({'CSM'},(length(data_mild_csm)+length(data_mod_csm)),1)]);

group_id = categorical([repmat({'0'},length(data_control),1);repmat({'1'},(length(data_mild_csm)+length(data_mod_csm)),1)]);

terminal2 = strcat('all_patients_all_features','_data.csv');
table_out=table(group,group_id,dti_adc,dti_axial,dti_fa,dti_radial,fiber_axial,fiber_fa,...
    fiber_radial,fiber_fraction,hindered_fraction,restricted_fraction,water_fraction);

table_out.Properties.VariableNames{1} = 'Group';
table_out.Properties.VariableNames{2} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs all CSM by group

group = categorical([repmat({'Control'},length(data_control),1);repmat({'Mild_CSM'},length(data_mild_csm),1);repmat({'Moderate_CSM'},length(data_mod_csm),1)]);

group_id = categorical([repmat({'0'},length(data_control),1);repmat({'1'},length(data_mild_csm),1);repmat({'2'},length(data_mod_csm),1)]);

terminal2 = strcat('all_patients_all_features_by_CSM_group','_data.csv');
table_out=table(group,group_id,dti_adc,dti_axial,dti_fa,dti_radial,fiber_axial,fiber_fa,...
    fiber_radial,fiber_fraction,hindered_fraction,restricted_fraction,water_fraction);

table_out.Properties.VariableNames{1} = 'Group';
table_out.Properties.VariableNames{2} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs Mild CSM

terminal2 = strcat('all_patients_all_features_mild_CSM','_data.csv');
table_mild = table_out;
table_mild((length(data_control)+length(data_mild_csm)+1:end),:)=[];

writetable(table_mild,fullfile(out_dir,terminal2));

%Controls vs moderate CSM 

group_id = categorical([repmat({'0'},length(data_control),1);repmat({'1'},length(data_mod_csm),1)]);
terminal2 = strcat('all_patients_all_features_moderate_CSM','_data.csv');
table_moderate = table_out;
table_moderate((length(data_control)+1:(length(data_control)+length(data_mild_csm))),:)=[];
table_moderate(:,2) = table(group_id);
table_moderate.Properties.VariableNames{2} = 'Group_ID';

writetable(table_moderate,fullfile(out_dir,terminal2));

%Mild vs moderate CSM 

group_id = categorical([repmat({'0'},length(data_mild_csm),1);repmat({'1'},length(data_mod_csm),1)]);
terminal2 = strcat('all_patients_all_features_only_CSM','_data.csv');
table_csm = table_out;
table_csm((1:length(data_control)),:)=[];
table_csm(:,2) = table(group_id);
table_csm.Properties.VariableNames{2} = 'Group_ID';

writetable(table_csm,fullfile(out_dir,terminal2));
