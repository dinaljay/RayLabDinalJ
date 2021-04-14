% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROIs

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

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);

dhi_features = ["DTI ADC";"DTI Axial";"DTI FA";"DTI Radial";"Fiber Axial";"Fiber FA ";...
    "Fiber Radial";"Fiber Fraction";"Hindered Fraction";"Restricted Fraction";"Water Fraction";"Axon Volume";"Inflammation Volume"];

%% Create variable stores
% %B0 Map
%
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_dti_adc_map_data.mat');
all_control{1,1} = data_control;
all_csm{1,1} = data_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_dti_axial_map_data.mat');
all_control{2,1} = data_control;
all_csm{2,1} = data_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_dti_fa_map_data.mat');
all_control{3,1} = data_control;
all_csm{3,1} = data_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_dti_radial_map_data.mat');
all_control{4,1} = data_control;
all_csm{4,1} = data_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_fiber1_axial_map_data.mat');
all_control{5,1} = data_control;
all_csm{5,1} = data_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_fiber1_fa_map_data.mat');
all_control{6,1} = data_control;
all_csm{6,1} = data_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_fiber1_radial_map_data.mat');
all_control{7,1} = data_control;
all_csm{7,1} = data_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_fiber_fraction_map_data.mat');
all_control{8,1} = data_control;
all_csm{8,1} = data_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_hindered_fraction_map_data.mat');
all_control{9,1} = data_control;
all_csm{9,1} = data_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_restricted_fraction_map_data.mat');
all_control{10,1} = data_control;
all_csm{10,1} = data_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_water_fraction_map_data.mat');
all_control{11,1} = data_control;
all_csm{11,1} = data_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_axon_volume_data.mat');
all_control{12,1} = control_axon_volume;
all_csm{12,1} = csm_axon_volume;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/ROI/CSM/csm_inflammation_volume_data.mat');
all_control{13,1} = control_inflammation_volume;
all_csm{13,1} = csm_inflammation_volume;

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


