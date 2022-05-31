% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
% mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50]; %Dinal's classification
%CSM_P01 template no good
mild_cm_subjects = [3,4,15,16,18,19,21,23,24,28,29,32,36,38,40,42,43,44,45,46,48,49,50]; %Justin's classification

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
% moderate_cm_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41]; %Dinal's classification

moderate_cm_subjects = [2,5,6,9,10,11,12,13,14,20,22,25,26,27,30,31,34,37,41]; %Justin's classification

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);
% slices = [4];

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...
    "fiber1_extra_axial_map";"fiber1_extra_fraction_map";"fiber1_extra_radial_map";"fiber1_intra_axial_map";"fiber1_intra_fraction_map";...
    "fiber1_intra_radial_map";"fiber1_rgba_map_itk";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_extra_axial_map";...
    "fiber2_extra_fraction_map";"fiber2_extra_radial_map";"fiber2_intra_axial_map";"fiber2_intra_fraction_map";"fiber2_intra_radial_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

in_dir = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA';

%% Create variable stores

for n=1:numel(dhi_features)
    
    %Pre-op data
    file_name = strcat('control_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI/Control/',file_name);
    load(temp);
    all_control_pre{n,1} = data_control;
    
    file_name = strcat('mild_csm_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI/Mild_CSM/',file_name);
    load(temp);
    mild_csm_pre{n,1} = data_mild_csm;
    
    file_name = strcat('mod_csm_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI/Moderate_CSM/',file_name);
    load(temp);
    mod_csm_pre{n,1} = data_mod_csm;
    
    clear data_control; clear data_mild_csm; clear data_mod_csm; clear temp;
    
end

%% Create patient IDs
x=1;
for k = 1:numel(controls)
    
    controlID{x,1} = strcat('CSM_C0',num2str(controls(k)));
    x=x+1;
end

x=1;
for k = 1:numel(mild_cm_subjects)
    
    mildID{x,1} = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
    x=x+1;
end

x=1;
for k = 1:numel(moderate_cm_subjects)
    
    modID{x,1} = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
    x=x+1;
end

%% Generate csv files

var_Names = ["Patient_ID", "Group", "Group_ID", "b0_map", "dti_adc_map", "dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_fa_map", "dti_g_map",...
                    "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_dirx_map", "fiber1_diry_map", "fiber1_dirz_map",...
                    "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",...
                    "fiber1_intra_radial_map", "fiber1_rgba_map_itk", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_extra_axial_map",...
                    "fiber2_extra_fraction_map", "fiber2_extra_radial_map", "fiber2_intra_axial_map", "fiber2_intra_fraction_map", "fiber2_intra_radial_map",...
                    "fraction_rgba_map", "hindered_adc_map", "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map",...
                    "water_adc_map", "water_fraction_map"];

% Control pre_op
control_pre = length(all_control_pre{1,1});
group = categorical([repmat({'Control'},control_pre,1)]);
group_id = categorical([repmat({'0'},control_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data(:,i) = cell2mat(all_control_pre{i,1});
    
end

t1 = table(controlID, group, group_id);
t2 = array2table(data);
t_control_pre = [t1,t2];
t_control_pre = renamevars(t_control_pre,1:width(t_control_pre),var_Names);
clear group; clear group_id; clear data; clear t1; clear t2;

%CSM pre_op

csm_pre = length(mild_csm_pre{1,1})+length(mod_csm_pre{1,1});
group = categorical([repmat({'CSM'},csm_pre,1)]);
group_id = categorical([repmat({'1'},csm_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data1(:,i) = cell2mat(mild_csm_pre{i,1});

end

for i=1:length(dhi_features)
    
    data2(:,i) = cell2mat(mod_csm_pre{i,1});

end

csmID = [mildID; modID];
t1 = table(csmID, group, group_id);
t2 = array2table([data1; data2]);
t_csm_pre = [t1,t2];
t_csm_pre = renamevars(t_csm_pre,1:width(t_csm_pre),var_Names);
clear group; clear group_id; clear data1; clear data2; clear t1; clear t2;

%Mild CSM pre_op

mild_pre = length(mild_csm_pre{1,1});
group = categorical([repmat({'Mild CSM'},mild_pre,1)]);
group_id = categorical([repmat({'1'},mild_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data(:,i) = cell2mat(mild_csm_pre{i,1});

end

t1 = table(mildID, group, group_id);
t2 = array2table(data);
t_mild_pre = [t1,t2];
t_mild_pre = renamevars(t_mild_pre,1:width(t_mild_pre),var_Names);
clear group; clear group_id; clear data; clear t1; clear t2;

%Moderate CSM pre_op

mod_pre = length(mod_csm_pre{1,1});
group = categorical([repmat({'Moderate CSM'},mod_pre,1)]);
group_id = categorical([repmat({'2'},mod_pre,1)]);
data = [];

for i=1:length(dhi_features)

    data(:,i) = cell2mat(mod_csm_pre{i,1});
    
end

t1 = table(modID, group, group_id);
t2 = array2table(data);
t_mod_pre = [t1,t2];
t_mod_pre = renamevars(t_mod_pre,1:width(t_mod_pre),var_Names);

clear group; clear group_id; clear data; clear t1; clear t2;

%% Save csv files

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI/DBSI_CSV_Data/Pre_op';

%Controls vs all CSM

table_out = [t_control_pre;t_csm_pre];

terminal2 = strcat('all_patients_all_features','_data.csv');
writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs all CSM by group

table_all = [t_control_pre;t_mild_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features_by_CSM_group','_data.csv');
writetable(table_all,fullfile(out_dir,terminal2));

%Controls vs Mild CSM

table_mild = [t_control_pre;t_mild_pre];

terminal2 = strcat('all_patients_all_features_mild_CSM','_data.csv');
writetable(table_mild,fullfile(out_dir,terminal2));

%% Group ID update

%Mild CSM pre_op

mild_pre = length(mild_csm_pre{1,1});
group = categorical([repmat({'Mild CSM'},mild_pre,1)]);
group_id = categorical([repmat({'0'},mild_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data(:,i) = cell2mat(mild_csm_pre{i,1});

end

t1 = table(mildID, group, group_id);
t2 = array2table(data);
t_mild_pre = [t1,t2];
t_mild_pre = renamevars(t_mild_pre,1:width(t_mild_pre),var_Names);
clear group; clear group_id; clear data; clear t1; clear t2;

%Moderate CSM pre_op

mod_pre = length(mod_csm_pre{1,1});
group = categorical([repmat({'Moderate CSM'},mod_pre,1)]);
group_id = categorical([repmat({'1'},mod_pre,1)]);
data = [];

for i=1:length(dhi_features)

    data(:,i) = cell2mat(mod_csm_pre{i,1});
    
end

t1 = table(modID, group, group_id);
t2 = array2table(data);
t_mod_pre = [t1,t2];
t_mod_pre = renamevars(t_mod_pre,1:width(t_mod_pre),var_Names);

clear group; clear group_id; clear data; clear t1; clear t2;

%% Resume file creation

%Controls vs moderate CSM 
table_moderate = [t_control_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features_moderate_CSM','_data.csv');
writetable(table_moderate,fullfile(out_dir,terminal2));

%Mild vs moderate CSM 

table_csm = [t_mild_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features_only_CSM','_data.csv');
writetable(table_csm,fullfile(out_dir,terminal2));
