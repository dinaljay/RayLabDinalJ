% This script calculates the axon and inflammation volume for each slice of
% each patient

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
% out_dir_csm_csv = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data/CSM';

% out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Control';
out_dir_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped';

% out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Mild_CSM';
% out_dir_moderate_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Moderate_CSM';


%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% awaiting confirmation on CSM subjects [1,11,]

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

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
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_volume_data.mat');
csm_volumes = cell2mat(csm_volumes);

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_fiber_fraction_map_data.mat');
csm_fiber_fraction = cell2mat(data_csm);

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_hindered_fraction_map_data.mat');
csm_hindered = cell2mat(data_csm);

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped/csm_restricted_fraction_map_data.mat');
csm_restricted = cell2mat(data_csm);

%% Volume Calculations

csm_inflammation = csm_hindered + csm_restricted;

for i = 1:numel(cm_subjects)
    for j = 1:numel(slices)
        csm_inflammation_volume{i,j} = csm_volumes(i,j)*csm_inflammation(i,j);
        csm_axon_volume{i,j} = csm_volumes(i,j)*csm_fiber_fraction(i,j);
    end
end

terminal = strcat('csm_inflammation_volume','_data.mat');
save(fullfile(out_dir_csm,terminal),'cm_subjects','mild_cm_subjects','moderate_cm_subjects','csm_inflammation_volume');

terminal = strcat('csm_axon_volume','_data.mat');
save(fullfile(out_dir_csm,terminal),'cm_subjects','mild_cm_subjects','moderate_cm_subjects','csm_axon_volume');

return;

%% Data saving

%Control Patients

terminal2 = strcat('control_inflammation_volume','_data.csv');
table_out=cell2table(control_inflammation_volume);
table_out.Properties.VariableNames{1} = 'Slice_1';
table_out.Properties.VariableNames{2} = 'Slice_2';
table_out.Properties.VariableNames{3} = 'Slice_3';
table_out.Properties.VariableNames{4} = 'Slice_4';
writetable(table_out,fullfile(out_dir_control_csv,terminal2))

terminal2 = strcat('control_axon_volume','_data.csv');
table_out=cell2table(control_axon_volume);
table_out.Properties.VariableNames{1} = 'Slice_1';
table_out.Properties.VariableNames{2} = 'Slice_2';
table_out.Properties.VariableNames{3} = 'Slice_3';
table_out.Properties.VariableNames{4} = 'Slice_4';
writetable(table_out,fullfile(out_dir_control_csv,terminal2))

%All Cervical Myelopathy Patients

terminal2 = strcat('csm_inflammation_volume','_data.csv');
table_out=cell2table(csm_inflammation_volume);
table_out.Properties.VariableNames{1} = 'Slice_1';
table_out.Properties.VariableNames{2} = 'Slice_2';
table_out.Properties.VariableNames{3} = 'Slice_3';
table_out.Properties.VariableNames{4} = 'Slice_4';
writetable(table_out,fullfile(out_dir_csm_csv,terminal2));

terminal2 = strcat('csm_axon_volume','_data.csv');
table_out=cell2table(csm_axon_volume);
table_out.Properties.VariableNames{1} = 'Slice_1';
table_out.Properties.VariableNames{2} = 'Slice_2';
table_out.Properties.VariableNames{3} = 'Slice_3';
table_out.Properties.VariableNames{4} = 'Slice_4';
writetable(table_out,fullfile(out_dir_csm_csv,terminal2));

%% All patients
out_dir_all = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_ROI/DBSI_CSV_Data/All_patients';

group_id = categorical([repmat({'Control'},numel(controls),1);repmat({'CSM'},numel(cm_subjects),1)]);

all_data = cell2mat([control_inflammation_volume;csm_inflammation_volume]);
all_data_1 = all_data(:,1);
all_data_2 = all_data(:,2);
all_data_3 = all_data(:,3);
all_data_4 = all_data(:,4);
terminal2 = strcat('all_inflammation_volume','_data.csv');
table_out=table(group_id,all_data_1,all_data_2,all_data_3,all_data_4);
table_out.Properties.VariableNames{1} = 'Group';
table_out.Properties.VariableNames{2} = 'Slice_1';
table_out.Properties.VariableNames{3} = 'Slice_2';
table_out.Properties.VariableNames{4} = 'Slice_3';
table_out.Properties.VariableNames{5} = 'Slice_4';
writetable(table_out,fullfile(out_dir_all,terminal2));

all_data = cell2mat([control_axon_volume;csm_axon_volume]);
all_data_1 = all_data(:,1);
all_data_2 = all_data(:,2);
all_data_3 = all_data(:,3);
all_data_4 = all_data(:,4);
terminal2 = strcat('all_axon_volume','_data.csv');
table_out=table(group_id,all_data_1,all_data_2,all_data_3,all_data_4);
table_out.Properties.VariableNames{1} = 'Group';
table_out.Properties.VariableNames{2} = 'Slice_1';
table_out.Properties.VariableNames{3} = 'Slice_2';
table_out.Properties.VariableNames{4} = 'Slice_3';
table_out.Properties.VariableNames{5} = 'Slice_4';
writetable(table_out,fullfile(out_dir_all,terminal2));
