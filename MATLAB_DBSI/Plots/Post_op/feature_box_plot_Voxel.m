% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on voxels

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["DTI ADC";"DTI Axial";"DTI FA";"Fiber Axial";"Fiber FA ";...
    "Fiber Radial";"Fiber Fraction";"Hindered Fraction";"Restricted Fraction";"Water Fraction";"Axon Volume";"Inflammation Volume"];

%% Create variable stores
% %B0 Map
%
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_adc_map_data.mat');
all_control{1,1} = data_control;
all_csm{1,1} = data_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_axial_map_data.mat');
all_control{2,1} = data_control;
all_csm{2,1} = data_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_fa_map_data.mat');
all_control{3,1} = data_control;
all_csm{3,1} = data_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_axial_map_data.mat');
all_control{4,1} = data_control;
all_csm{4,1} = data_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_fa_map_data.mat');
all_control{5,1} = data_control;
all_csm{5,1} = data_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_radial_map_data.mat');
all_control{6,1} = data_control;
all_csm{6,1} = data_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber_fraction_map_data.mat');
all_control{7,1} = data_control;
all_csm{7,1} = data_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_hindered_fraction_map_data.mat');
all_control{8,1} = data_control;
all_csm{8,1} = data_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_restricted_fraction_map_data.mat');
all_control{9,1} = data_control;
all_csm{9,1} = data_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_water_fraction_map_data.mat');
all_control{10,1} = data_control;
all_csm{10,1} = data_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_axon_volume_data.mat');
all_control{11,1} = control_axon_volume;
all_csm{11,1} = csm_axon_volume;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_inflammation_volume_data.mat');
all_control{12,1} = control_inflammation_volume;
all_csm{12,1} = csm_inflammation_volume;

%% Plots

for i = 1:numel(dhi_features)
    figure
    for j = 1:numel(slices)
        slice_num = strcat("Slice ",num2str(slices(j)));
       
        x1 = cell2mat(all_control{i,1}(:,j));
        x2 = cell2mat(all_csm{i,1}(:,j));
        x = [x1; x2];
        g = [zeros(size(x1));ones(size(x2))];
        
        subplot(2,2,j)
        boxplot(x,g,'Notch','on','Labels',{'Controls','CSM'},'Whisker',1)
        temp = strcat(dhi_features(i)," - ",slice_num);
        title(sprintf('%s',temp))
        
    end
end


