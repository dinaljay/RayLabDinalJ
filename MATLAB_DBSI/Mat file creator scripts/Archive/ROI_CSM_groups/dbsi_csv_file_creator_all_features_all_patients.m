% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% awaiting confirmation on CSM subjects [1,11,]

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"dti_radial_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map";...
    "axon_volume";"inflammation_volume"];

%% Create variable stores
%B0 Map

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_b0_map_data.mat');
b0 = [cell2mat(data_control);cell2mat(data_csm)];
b0_1 = b0(:,1);
b0_2 = b0(:,2);
b0_3 = b0(:,3);
b0_4 = b0(:,4);

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_control);cell2mat(data_csm)];
dti_adc_1 = dti_adc(:,1);
dti_adc_2 = dti_adc(:,2);
dti_adc_3 = dti_adc(:,3);
dti_adc_4 = dti_adc(:,4);

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_control);cell2mat(data_csm)];
dti_axial_1 = dti_axial(:,1);
dti_axial_2 = dti_axial(:,2);
dti_axial_3 = dti_axial(:,3);
dti_axial_4 = dti_axial(:,4);

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_control);cell2mat(data_csm)];
dti_fa_1 = dti_fa(:,1);
dti_fa_2 = dti_fa(:,2);
dti_fa_3 = dti_fa(:,3);
dti_fa_4 = dti_fa(:,4);

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM/csm_dti_radial_map_data.mat');
dti_radial = [cell2mat(data_control);cell2mat(data_csm)];
dti_radial_1 = dti_radial(:,1);
dti_radial_2 = dti_radial(:,2);
dti_radial_3 = dti_radial(:,3);
dti_radial_4 = dti_radial(:,4);

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_control);cell2mat(data_csm)];
fiber_axial_1 = fiber_axial(:,1);
fiber_axial_2 = fiber_axial(:,2);
fiber_axial_3 = fiber_axial(:,3);
fiber_axial_4 = fiber_axial(:,4);

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_control);cell2mat(data_csm)];
fiber_fa_1 = fiber_fa(:,1);
fiber_fa_2 = fiber_fa(:,2);
fiber_fa_3 = fiber_fa(:,3);
fiber_fa_4 = fiber_fa(:,4);

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_control);cell2mat(data_csm)];
fiber_radial_1 = fiber_radial(:,1);
fiber_radial_2 = fiber_radial(:,2);
fiber_radial_3 = fiber_radial(:,3);
fiber_radial_4 = fiber_radial(:,4);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_control);cell2mat(data_csm)];
fiber_fraction_1 = fiber_fraction(:,1);
fiber_fraction_2 = fiber_fraction(:,2);
fiber_fraction_3 = fiber_fraction(:,3);
fiber_fraction_4 = fiber_fraction(:,4);

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_control);cell2mat(data_csm)];
hindered_fraction_1 = hindered_fraction(:,1);
hindered_fraction_2 = hindered_fraction(:,2);
hindered_fraction_3 = hindered_fraction(:,3);
hindered_fraction_4 = hindered_fraction(:,4);

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_control);cell2mat(data_csm)];
restricted_fraction_1 = restricted_fraction(:,1);
restricted_fraction_2 = restricted_fraction(:,2);
restricted_fraction_3 = restricted_fraction(:,3);
restricted_fraction_4 = restricted_fraction(:,4);

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_control);cell2mat(data_csm)];
water_fraction_1 = water_fraction(:,1);
water_fraction_2 = water_fraction(:,2);
water_fraction_3 = water_fraction(:,3);
water_fraction_4 = water_fraction(:,4);

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_axon_volume_data.mat');
axon_volume = [cell2mat(control_axon_volume);cell2mat(csm_axon_volume)];
axon_volume_1 = axon_volume(:,1);
axon_volume_2 = axon_volume(:,2);
axon_volume_3 = axon_volume(:,3);
axon_volume_4 = axon_volume(:,4);

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_inflammation_volume_data.mat');
inflammation_volume = [cell2mat(control_inflammation_volume);cell2mat(csm_inflammation_volume)];
inflammation_volume_1 = inflammation_volume(:,1);
inflammation_volume_2 = inflammation_volume(:,2);
inflammation_volume_3 = inflammation_volume(:,3);
inflammation_volume_4 = inflammation_volume(:,4);

%% Create patient IDs
x=1;
for k = 1:numel(controls)
    
    patientID{x,1} = strcat('CSM_C0',num2str(controls(k)));
    x=x+1;
end

for k = 1:numel(cm_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(cm_subjects(k)));
    x=x+1;
end

%% Save csv file

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data';

%Controls vs all CSM groups

group = categorical([repmat({'Control'},numel(controls),1);repmat({'Mild_CSM'},numel(mild_cm_subjects),1);repmat({'Moderate_CSM'},numel(moderate_cm_subjects),1)]);

group_id = categorical([repmat({'0'},numel(controls),1);repmat({'1'},numel(mild_cm_subjects),1);repmat({'2'},numel(moderate_cm_subjects),1)]);

terminal2 = strcat('all_patients_all_features_by_CSM_group','_data.csv');
table_out=table(patientID,group,group_id,dti_adc_1,dti_adc_2,dti_adc_3,dti_adc_4,...
    dti_axial_1,dti_axial_2,dti_axial_3,dti_axial_4,dti_fa_1,dti_fa_2,dti_fa_3,dti_fa_4,...
    fiber_axial_1,fiber_axial_2,fiber_axial_3,fiber_axial_4,fiber_fa_1,fiber_fa_2,fiber_fa_3,fiber_fa_4,dti_radial_1,dti_radial_2,dti_radial_3,dti_radial_4,...
    fiber_radial_1,fiber_radial_2,fiber_radial_3,fiber_radial_4,fiber_fraction_1,fiber_fraction_2,fiber_fraction_3,fiber_fraction_4,...
    hindered_fraction_1,hindered_fraction_2,hindered_fraction_3,hindered_fraction_4,restricted_fraction_1,restricted_fraction_2,restricted_fraction_3,restricted_fraction_4,...
    water_fraction_1,water_fraction_2,water_fraction_2,water_fraction_4,axon_volume_1,axon_volume_2,axon_volume_3,axon_volume_4,...
    inflammation_volume_1,inflammation_volume_2,inflammation_volume_3,inflammation_volume_4);

table_out.Properties.VariableNames{1} = 'Patient_ID';
table_out.Properties.VariableNames{2} = 'Group';
table_out.Properties.VariableNames{3} = 'Group_ID';

writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs all Mild CSM

terminal2 = strcat('all_patients_all_features_mild_CSM','_data.csv');
table_mild = table_out;
table_mild(((end-numel(moderate_cm_subjects)):end),:)=[];

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
table_csm((1:length(data_control)),:)=[];
table_csm(:,3) = table(group_id);
table_csm.Properties.VariableNames{3} = 'Group_ID';

writetable(table_csm,fullfile(out_dir,terminal2));

