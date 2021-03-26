% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

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

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"dti_radial_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map";...
    "axon_volume";"inflammation_volume"];

%% Create variable stores
%B0 Map

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_b0_map_data.mat');
b0 = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
dti_radial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_control);cell2mat(data_mild_csm);cell2mat(data_mod_csm)];
clear data_control; clear data_mild_csm; clear data_mod_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_axon_volume_data.mat');
axon_volume = [cell2mat(control_axon_volume);cell2mat(mild_csm_axon_volume);cell2mat(mod_csm_axon_volume)];
clear control_axon_volume; clear mild_csm_axon_volume; clear mod_csm_axon-volume;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
inflammation_volume = [cell2mat(control_inflammation_volume);cell2mat(mild_csm_inflammation_volume);cell2mat(mod_csm_inflammation_volume)];
clear control_inflammation_volume; clear mild_csm_inflammation_volume; clear mod_csm_inflammation_volume;

%% Create patient IDs
x=1;
for k = 1:numel(controls)
    
    patientID{x,1} = strcat('CSM_C0',num2str(controls(k)));
    x=x+1;
end

for k = 1:numel(mild_cm_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
    x=x+1;
end

for k = 1:numel(moderate_cm_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
    x=x+1;
end

%% Save csv file

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op';

%Controls vs all CSM

group = categorical([repmat({'Control'},numel(controls),1);repmat({'CSM'},(numel(mild_cm_subjects)+numel(moderate_cm_subjects)),1)]);

group_id = categorical([repmat({'0'},numel(controls),1);repmat({'1'},(numel(mild_cm_subjects)+numel(moderate_cm_subjects)),1)]);

terminal2 = strcat('all_patients_all_features','_data.csv');
table_out=table(patientID, group,group_id,dti_adc,dti_axial,dti_fa,dti_radial,fiber_axial,fiber_fa,...
    fiber_radial,fiber_fraction,hindered_fraction,restricted_fraction,water_fraction,axon_volume,inflammation_volume);

table_out.Properties.VariableNames{1} = 'Patient_ID';
table_out.Properties.VariableNames{2} = 'Group';
table_out.Properties.VariableNames{3} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs all CSM by group

group = categorical([repmat({'Control'},numel(controls),1);repmat({'Mild_CSM'},numel(mild_cm_subjects),1);repmat({'Moderate_CSM'},numel(moderate_cm_subjects),1)]);

group_id = categorical([repmat({'0'},numel(controls),1);repmat({'1'},numel(mild_cm_subjects),1);repmat({'2'},numel(moderate_cm_subjects),1)]);

terminal2 = strcat('all_patients_all_features_by_CSM_group','_data.csv');
table_out=table(patientID, group,group_id,dti_adc,dti_axial,dti_fa,dti_radial,fiber_axial,fiber_fa,...
    fiber_radial,fiber_fraction,hindered_fraction,restricted_fraction,water_fraction,axon_volume,inflammation_volume);

table_out.Properties.VariableNames{1} = 'Patient_ID';
table_out.Properties.VariableNames{2} = 'Group';
table_out.Properties.VariableNames{3} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs all Mild CSM

terminal2 = strcat('all_patients_all_features_mild_CSM','_data.csv');
table_mild = table_out;
table_mild(((end-numel(moderate_cm_subjects)+1):end),:)=[];

writetable(table_mild,fullfile(out_dir,terminal2));

%Controls vs all moderate CSM 

terminal2 = strcat('all_patients_all_features_moderate_CSM','_data.csv');
group_id = categorical([repmat({'0'},numel(controls),1);repmat({'1'},numel(moderate_cm_subjects),1)]);
table_moderate = table_out;
table_moderate((numel(controls)+1:(numel(controls)+numel(mild_cm_subjects))),:)=[];
table_moderate(:,3) = table(group_id);
table_moderate.Properties.VariableNames{3} = 'Group_ID';

writetable(table_moderate,fullfile(out_dir,terminal2));


%Mild vs moderate CSM 

group_id = categorical([repmat({'0'},numel(mild_cm_subjects),1);repmat({'1'},numel(moderate_cm_subjects),1)]);
terminal2 = strcat('all_patients_all_features_only_CSM','_data.csv');
table_csm = table_out;
table_csm((1:numel(controls)),:)=[];
table_csm(:,3) = table(group_id);
table_csm.Properties.VariableNames{3} = 'Group_ID';

writetable(table_csm,fullfile(out_dir,terminal2));
