% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_dirz_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_axial_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...    
    "fiber1_fa_map";"fiber1_fiber_fraction_map";"fiber1_radial_map";"fiber1_rgba_map";"fiber1_rgba_map_itk";...
    "fiber2_axial_map";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_fa_map";...
    "fiber2_fiber_fraction_map";"fiber2_radial_map";"fiber_fraction_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

in_dir = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI';

%% Create variable stores

for n=1:numel(dhi_features)
    
    %Pre-op data
    file_name = strcat('control_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI_Voxel/All_slices/Control/',file_name);
    load(temp);
    all_control_pre{n,1} = data_control;
    
    file_name = strcat('mild_csm_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI_Voxel/All_slices/Mild_CSM/',file_name);
    load(temp);
    mild_csm_pre{n,1} = data_mild_csm;
    
    file_name = strcat('mod_csm_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Pre_op/ROI_Voxel/All_slices/Moderate_CSM/',file_name);
    load(temp);
    mod_csm_pre{n,1} = data_mod_csm;
    
    clear data_control; clear data_mild_csm; clear data_mod_csm; clear temp;
    
end

%% Generate csv files

var_Names = ["Group", "Group_ID", "b0_map", "dti_adc_map","dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_dirz_map", "dti_fa_map",...
                "dti_g_map", "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_axial_map", "fiber1_dirx_map",...
                "fiber1_diry_map", "fiber1_dirz_map", "fiber1_fa_map", "fiber1_fiber_fraction_map", "fiber1_radial_map", "fiber1_rgba_map",...
                "fiber1_rgba_map_itk", "fiber2_axial_map", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_fa_map",...
                "fiber2_fiber_fraction_map", "fiber2_radial_map", "fiber_fraction_map", "fraction_rgba_map", "hindered_adc_map",...
                "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map", "water_adc_map",...
                "water_fraction_map"];

% Control pre_op
control_pre = length(all_control_pre{1,1});
group = categorical([repmat({'Control'},control_pre,1)]);
group_id = categorical([repmat({'0'},control_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data(:,i) = cell2mat(all_control_pre{i,1});
    
end

t1 = table(group, group_id);
t2 = array2table(data);
t_control_pre = [t1,t2];
t_control_pre = renamevars(t_control_pre,1:width(t_control_pre),var_Names);
clear group; clear group_id; clear data; clear t1; clear t2;

%Mild CSM pre_op

mild_pre = length(mild_csm_pre{1,1});
group = categorical([repmat({'Mild CSM'},mild_pre,1)]);
group_id = categorical([repmat({'1'},mild_pre,1)]);
data = [];

for i=1:length(dhi_features)
    
    data(:,i) = cell2mat(mild_csm_pre{i,1});

end

t1 = table(group, group_id);
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

t1 = table(group, group_id);
t2 = array2table(data);
t_mod_pre = [t1,t2];
t_mod_pre = renamevars(t_mod_pre,1:width(t_mod_pre),var_Names);

clear group; clear group_id; clear data; clear t1; clear t2;

%% Save csv files

out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_Voxel/Pre_op';

%Controls vs all CSM

table_out = [t_control_pre;t_mild_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features','_data.csv');
writetable(table_out,fullfile(out_dir,terminal2));

%Controls vs Mild CSM

table_mild = [t_control_pre;t_mild_pre];

terminal2 = strcat('all_patients_all_features_mild_CSM','_data.csv');
writetable(table_mild,fullfile(out_dir,terminal2));

%Controls vs moderate CSM 

table_moderate = [t_control_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features_moderate_CSM','_data.csv');
writetable(table_moderate,fullfile(out_dir,terminal2));

%Mild vs moderate CSM 

table_csm = [t_mild_pre;t_mod_pre];

terminal2 = strcat('all_patients_all_features_only_CSM','_data.csv');
writetable(table_csm,fullfile(out_dir,terminal2));
