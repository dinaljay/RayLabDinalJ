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
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Pre_op/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/Pre_op/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
all_control{1,1} = data_control;
mild_csm{1,1} = data_mild_csm;
mod_csm{1,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
all_control{2,1} = data_control;
mild_csm{2,1} = data_mild_csm;
mod_csm{2,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
all_control{3,1} = data_control;
mild_csm{3,1} = data_mild_csm;
mod_csm{3,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
all_control{4,1} = data_control;
mild_csm{4,1} = data_mild_csm;
mod_csm{4,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
all_control{5,1} = data_control;
mild_csm{5,1} = data_mild_csm;
mod_csm{5,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
all_control{6,1} = data_control;
mild_csm{6,1} = data_mild_csm;
mod_csm{6,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
all_control{7,1} = data_control;
mild_csm{7,1} = data_mild_csm;
mod_csm{7,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
all_control{8,1} = data_control;
mild_csm{8,1} = data_mild_csm;
mod_csm{8,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
all_control{9,1} = data_control;
mild_csm{9,1} = data_mild_csm;
mod_csm{9,1} = data_mod_csm;;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
all_control{10,1} = data_control;
mild_csm{10,1} = data_mild_csm;
mod_csm{10,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
all_control{11,1} = data_control;
mild_csm{11,1} = data_mild_csm;
mod_csm{11,1} = data_mod_csm;

clear data_control; clear data_mild_csm; clear data_mod_csm;

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_axon_volume_data.mat');
all_control{12,1} = control_axon_volume;
mild_csm{12,1} = mild_csm_axon_volume;
mod_csm{12,1} = mod_csm_axon_volume;

clear control_axon_volume; clear mild_csm_axon_volume; clear mod_csm_axon_volume;

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Mild_CSM/mild_csm_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Pre_op/ROI/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
all_control{13,1} = control_inflammation_volume;
mild_csm{13,1} = mild_csm_inflammation_volume;
mod_csm{13,1} = mod_csm_inflammation_volume;

clear control_inflammation_volume; clear mild_csm_inflammation_volume; clear mod_csm_inflammation_volume;

%% Plots

colors = [1, 0, 0; 0, 1, 0; 0, 0, 1];

for i = 1:numel(dhi_features)
    figure
    
    x1 = cell2mat(all_control{i,1});
    x2 = cell2mat(mild_csm{i,1});
    x3 = cell2mat(mod_csm{i,1});
    x = [x1; x2; x3];
    
    g1 = repmat({'Control'},length(x1),1);
    g2 = repmat({'Mild'},length(x2),1);
    g3 = repmat({'Mod'},length(x3),1);
    g = [g1; g2; g3];
    
    boxplot(x,g,'Notch', 'on','Labels',{'Controls','Mild CSM', 'Moderate CSM'},'Whisker',0.5)
    temp = strcat(dhi_features(i),"");
    title(sprintf('%s',temp))
    
    h = findobj(gca,'Tag','Box');
    for j=1:length(h)
        patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
    end
    
end


