% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROIs for improved and non improved subjects

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...
    "fiber1_extra_axial_map";"fiber1_extra_fraction_map";"fiber1_extra_radial_map";"fiber1_intra_axial_map";"fiber1_intra_fraction_map";...
    "fiber1_intra_radial_map";"fiber1_rgba_map_itk";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_extra_axial_map";...
    "fiber2_extra_fraction_map";"fiber2_extra_radial_map";"fiber2_intra_axial_map";"fiber2_intra_fraction_map";"fiber2_intra_radial_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

in_dir = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA';

%% Create variable stores for Pre-op

for n=1:numel(dhi_features)
    
    %Pre-op data
    file_name = strcat('improv_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Improved_vs_Nonimproved/Pre_op/ROI/Improved/',file_name);
    load(temp);
    all_improv_pre{n,1} = data_improv;
    
    file_name = strcat('non_improv_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/',file_name);
    load(temp);
    all_non_improv_pre{n,1} = data_non_improv;
    
    clear data_improv; clear data_non_improv; clear temp;
    
end

%% Generate csv files

% Improved pre_op
improv_pre = length(all_improv_pre{1,1})*numel(dhi_features);
group = categorical([repmat({'Improved'},improv_pre,1)]);
operation = categorical([repmat({'Pre_op'},improv_pre,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(all_improv_pre{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(all_improv_pre{i,1})];
    data = [data;temp];
    
end

t_improv_pre = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Non Improved pre_op

non_improv_pre = length(all_non_improv_pre{1,1})*numel(dhi_features);
group = categorical([repmat({'Non-Improved'},non_improv_pre,1)]);
operation = categorical([repmat({'Pre_op'},non_improv_pre,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(all_non_improv_pre{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(all_non_improv_pre{i,1})];
    data = [data;temp];
    
end

t_non_improv_pre = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Final table

t_fin = [t_improv_pre;t_non_improv_pre];

%% Save table 

out_dir = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_improv_vs_nonimprov";
terminal = strcat('all_ROI','_data.csv');
writetable(t_fin,fullfile(out_dir,terminal));

