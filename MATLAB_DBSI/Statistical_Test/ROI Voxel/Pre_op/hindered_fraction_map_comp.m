%This script does a non-parametric comparison of hindered fraction 
% coefficients for each slice between CSM and control  patients

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB Scripts'));

%% Import data

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_hindered_fraction_map_data.mat');

%% Initialize variables

slices = (1:1:4);

%% Average data in a slice for each group

%Controls
for i = 1:numel(controls)
    for j = 1:numel(slices)        
        data_control_avg(i,j) = mean(data_control{i,j});
    end
    
end

%CM Patients

for i = 1:numel(cm_subjects)
    for j = 1:numel(slices)       
        data_csm_avg(i,j) = mean(data_csm{i,j});
    end
    
end


%% U test implementation

for i = 1:numel(slices)
    
    cm = data_csm_avg(:,i);
    control = data_control_avg(:,i);
    
    [p_cm(i,1),~,stats_array_cm] = ranksum(cm,control,'method','approximate');
    zval_cm(i,1) = stats_array_cm.zval;

end