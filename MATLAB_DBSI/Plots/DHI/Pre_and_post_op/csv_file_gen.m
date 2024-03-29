% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROIs

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


%% Create variable stores for Pre-op
% %B0 Map
%
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI/Pre_op/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI/Pre_op/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
all_control_pre{1,1} = data_control;
mild_csm_pre{1,1} = data_mild_csm;
mod_csm_pre{1,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
all_control_pre{2,1} = data_control;
mild_csm_pre{2,1} = data_mild_csm;
mod_csm_pre{2,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
all_control_pre{3,1} = data_control;
mild_csm_pre{3,1} = data_mild_csm;
mod_csm_pre{3,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
all_control_pre{4,1} = data_control;
mild_csm_pre{4,1} = data_mild_csm;
mod_csm_pre{4,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
all_control_pre{5,1} = data_control;
mild_csm_pre{5,1} = data_mild_csm;
mod_csm_pre{5,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
all_control_pre{6,1} = data_control;
mild_csm_pre{6,1} = data_mild_csm;
mod_csm_pre{6,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
all_control_pre{7,1} = data_control;
mild_csm_pre{7,1} = data_mild_csm;
mod_csm_pre{7,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
all_control_pre{8,1} = data_control;
mild_csm_pre{8,1} = data_mild_csm;
mod_csm_pre{8,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
all_control_pre{9,1} = data_control;
mild_csm_pre{9,1} = data_mild_csm;
mod_csm_pre{9,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
all_control_pre{10,1} = data_control;
mild_csm_pre{10,1} = data_mild_csm;
mod_csm_pre{10,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
all_control_pre{11,1} = data_control;
mild_csm_pre{11,1} = data_mild_csm;
mod_csm_pre{11,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_axon_volume_data.mat');
all_control_pre{12,1} = control_axon_volume;
mild_csm_pre{12,1} = mild_csm_axon_volume;
mod_csm_pre{12,1} = mod_csm_axon_volume;

clear data_control; clear data_mild_csm; clear data_mod_csm;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM/mild_csm_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
all_control_pre{13,1} = control_inflammation_volume;
mild_csm_pre{13,1} = mild_csm_inflammation_volume;
mod_csm_pre{13,1} = mod_csm_inflammation_volume;

clear data_control; clear data_mild_csm; clear data_mod_csm;

%% Create variable stores for Post-op
% %B0 Map
%
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI/Post_op/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI/Post_op/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
all_control_post{1,1} = data_control;
mild_csm_post{1,1} = data_mild_csm;
mod_csm_post{1,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
all_control_post{2,1} = data_control;
mild_csm_post{2,1} = data_mild_csm;
mod_csm_post{2,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
all_control_post{3,1} = data_control;
mild_csm_post{3,1} = data_mild_csm;
mod_csm_post{3,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
all_control_post{4,1} = data_control;
mild_csm_post{4,1} = data_mild_csm;
mod_csm_post{4,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
all_control_post{5,1} = data_control;
mild_csm_post{5,1} = data_mild_csm;
mod_csm_post{5,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
all_control_post{6,1} = data_control;
mild_csm_post{6,1} = data_mild_csm;
mod_csm_post{6,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
all_control_post{7,1} = data_control;
mild_csm_post{7,1} = data_mild_csm;
mod_csm_post{7,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
all_control_post{8,1} = data_control;
mild_csm_post{8,1} = data_mild_csm;
mod_csm_post{8,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
all_control_post{9,1} = data_control;
mild_csm_post{9,1} = data_mild_csm;
mod_csm_post{9,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
all_control_post{10,1} = data_control;
mild_csm_post{10,1} = data_mild_csm;
mod_csm_post{10,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
all_control_post{11,1} = data_control;
mild_csm_post{11,1} = data_mild_csm;
mod_csm_post{11,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_axon_volume_data.mat');
all_control_post{12,1} = control_axon_volume;
mild_csm_post{12,1} = mild_csm_axon_volume;
mod_csm_post{12,1} = mod_csm_axon_volume;

clear data_control; clear data_mild_csm; clear data_mod_csm;


%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Mild_CSM/mild_csm_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
all_control_post{13,1} = control_inflammation_volume;
mild_csm_post{13,1} = mild_csm_inflammation_volume;
mod_csm_post{13,1} = mod_csm_inflammation_volume;

clear data_control; clear data_mild_csm; clear data_mod_csm;


%% Generate csv files

% Control pre_op
control_pre = length(all_control_pre{1,1})*numel(dhi_features);
group = categorical([repmat({'Control'},control_pre,1)]);
operation = categorical([repmat({'Pre_op'},control_pre,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(all_control_pre{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(all_control_pre{i,1})];
    data = [data;temp];
    
end

t_control_pre = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

% Control post_op
control_post = length(all_control_post{1,1})*numel(dhi_features);
group = categorical([repmat({'Control'},control_post,1)]);
operation = categorical([repmat({'Post_op'},control_post,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(all_control_post{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(all_control_post{i,1})];
    data = [data;temp];
    
end

t_control_post = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Mild CSM pre_op

mild_pre = length(mild_csm_pre{1,1})*numel(dhi_features);
group = categorical([repmat({'Mild CSM'},mild_pre,1)]);
operation = categorical([repmat({'Pre_op'},mild_pre,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(mild_csm_pre{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(mild_csm_pre{i,1})];
    data = [data;temp];
    
end

t_mild_pre = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Mild CSM post_op

mild_post = length(mild_csm_post{1,1})*numel(dhi_features);
group = categorical([repmat({'Mild CSM'},mild_post,1)]);
operation = categorical([repmat({'Post_op'},mild_post,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(mild_csm_post{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(mild_csm_post{i,1})];
    data = [data;temp];
    
end

t_mild_post = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Moderate CSM pre_op

mod_pre = length(mod_csm_pre{1,1})*numel(dhi_features);
group = categorical([repmat({'Moderate CSM'},mod_pre,1)]);
operation = categorical([repmat({'Pre_op'},mod_pre,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(mod_csm_pre{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(mod_csm_pre{i,1})];
    data = [data;temp];
    
end

t_mod_pre = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Moderate CSM post_op

mod_post = length(mod_csm_post{1,1})*numel(dhi_features);
group = categorical([repmat({'Moderate CSM'},mod_post,1)]);
operation = categorical([repmat({'Post_op'},mod_post,1)]);
feature_col = [];
data = [];
for i=1:length(dhi_features)
    
    feature = repmat(dhi_features(i),length(mod_csm_post{1,1}),1);
    feature_col = [feature_col;feature];
    
    temp = [(mod_csm_post{i,1})];
    data = [data;temp];
    
end

t_mod_post = table(feature_col, operation, group, data);
clear group; clear operation; clear feature_col; clear data;

%Final table

t_fin = [t_control_pre;t_control_post;t_mild_pre;t_mild_post;t_mod_pre;t_mod_post];

%% Save table 

out_dir = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI";
terminal = strcat('all_ROI','_data.csv');
writetable(t_fin,fullfile(out_dir,terminal));



