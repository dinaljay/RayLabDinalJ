% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROIs for improved and non improved subjects

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["DTI ADC";"DTI Axial";"DTI FA";"DTI Radial";"Fiber Axial";"Fiber FA ";...
    "Fiber Radial";"Fiber Fraction";"Hindered Fraction";"Restricted Fraction";"Water Fraction"];

%% Create variable stores for Pre-op

%B0 map
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_b0_map_data.mat');
% b0 = [cell2mat(data_improv);cell2mat(data_non_improv)];
% clear data_improv; clear data_non_improv;

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_adc_map_data.mat');

all_improv_pre{1,1} = data_improv;
all_non_improv_pre{1,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_axial_map_data.mat');

all_improv_pre{2,1} = data_improv;
all_non_improv_pre{2,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_fa_map_data.mat');

all_improv_pre{3,1} = data_improv;
all_non_improv_pre{3,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_dti_radial_map_data.mat');

all_improv_pre{4,1} = data_improv;
all_non_improv_pre{4,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_axial_map_data.mat');

all_improv_pre{5,1} = data_improv;
all_non_improv_pre{5,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_fa_map_data.mat');

all_improv_pre{6,1} = data_improv;
all_non_improv_pre{6,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber1_radial_map_data.mat');

all_improv_pre{7,1} = data_improv;
all_non_improv_pre{7,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_fiber_fraction_map_data.mat');

all_improv_pre{8,1} = data_improv;
all_non_improv_pre{8,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_hindered_fraction_map_data.mat');

all_improv_pre{9,1} = data_improv;
all_non_improv_pre{9,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_restricted_fraction_map_data.mat');

all_improv_pre{10,1} = data_improv;
all_non_improv_pre{10,1} = data_non_improv;

clear data_improv; clear data_non_improv;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved/improv_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved/non_improv_water_fraction_map_data.mat');

all_improv_pre{11,1} = data_improv;
all_non_improv_pre{11,1} = data_non_improv;

clear data_improv; clear data_non_improv;

%% Generate csv files

% Improv pre_op
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

out_dir = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_improv_vs_nonimprov";
terminal = strcat('all_ROI','_data.csv');
writetable(t_fin,fullfile(out_dir,terminal));

