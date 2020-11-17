% This script does a multinomial logistic regression using each DBSI feature to predict
% for patient group

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

controls = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];

moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Create variable stores
%B0 Map

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_b0_map_data.mat');
b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_control);cell2mat(data_csm)];

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_control);cell2mat(data_csm)];

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_control);cell2mat(data_csm)];

%Axon volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_axon_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_axon_volume_data.mat');
axon_volumes = [cell2mat(control_axon_volume);cell2mat(csm_axon_volume)];

%Inflammation volume
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Control/control_inflammation_volume_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/CSM/csm_inflammation_volume_data.mat');
inflammation_volumes = [cell2mat(control_inflammation_volume);cell2mat(csm_inflammation_volume)];

%% Logistic regression

control_group = categorical(zeros(numel(controls),1));
csm_group = categorical(ones(numel(cm_subjects),1));
all_group_ids = [control_group;csm_group];

% DTI_ADC Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(dti_adc(:,i),all_group_ids);
    
    csm_tstat_val(i,1) = stats.t(2,1);  % tstat Value
    csm_p_val(i,1) = stats.p(2,1);  % p Value
    csm_se_val(i,1) = stats.se(2,1);  %S E Value
    
end

% DTI Axial Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(dti_axial(:,i),all_group_ids);
    
    csm_tstat_val(i,2) = stats.t(2,1);  % tstat Value
    csm_p_val(i,2) = stats.p(2,1);  % p Value
    csm_se_val(i,2) = stats.se(2,1);  %S E Value
    
end

% DTI FA Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(dti_fa(:,i),all_group_ids);
    
    csm_tstat_val(i,3) = stats.t(2,1);  % tstat Value
    csm_p_val(i,3) = stats.p(2,1);  % p Value
    csm_se_val(i,3) = stats.se(2,1);  %S E Value
    
end

% Fiber Axial Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(fiber_axial(:,i),all_group_ids);
    
    csm_tstat_val(i,4) = stats.t(2,1);  % tstat Value
    csm_p_val(i,4) = stats.p(2,1);  % p Value
    csm_se_val(i,4) = stats.se(2,1);  %S E Value
    
end

% Fiber FA Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(fiber_fa(:,i),all_group_ids);
    
    csm_tstat_val(i,5) = stats.t(2,1);  % tstat Value
    csm_p_val(i,5) = stats.p(2,1);  % p Value
    csm_se_val(i,5) = stats.se(2,1);  %S E Value
    
end

% Fiber Radial Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(fiber_radial(:,i),all_group_ids);
    
    csm_tstat_val(i,6) = stats.t(2,1);  % tstat Value
    csm_p_val(i,6) = stats.p(2,1);  % p Value
    csm_se_val(i,6) = stats.se(2,1);  %S E Value
    
end

% Fiber Fraction Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(fiber_fraction(:,i),all_group_ids);
    
    csm_tstat_val(i,7) = stats.t(2,1);  % tstat Value
    csm_p_val(i,7) = stats.p(2,1);  % p Value
    csm_se_val(i,7) = stats.se(2,1);  %S E Value
    
end

% Hindered Fraction Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(hindered_fraction(:,i),all_group_ids);
    
    csm_tstat_val(i,8) = stats.t(2,1);  % tstat Value
    csm_p_val(i,8) = stats.p(2,1);  % p Value
    csm_se_val(i,8) = stats.se(2,1);  %S E Value
    
end

% Restricted Fraction Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(restricted_fraction(:,i),all_group_ids);
    
    csm_tstat_val(i,9) = stats.t(2,1);  % tstat Value
    csm_p_val(i,9) = stats.p(2,1);  % p Value
    csm_se_val(i,9) = stats.se(2,1);  %S E Value
    
end

% Water Fraction Map
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(water_fraction(:,i),all_group_ids);
    
    csm_tstat_val(i,10) = stats.t(2,1);  % tstat Value
    csm_p_val(i,10) = stats.p(2,1);  % p Value
    csm_se_val(i,10) = stats.se(2,1);  %S E Value
    
end

% Axon Volume
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(axon_volumes(:,i),all_group_ids);
    
    csm_tstat_val(i,11) = stats.t(2,1);  % tstat Value
    csm_p_val(i,11) = stats.p(2,1);  % p Value
    csm_se_val(i,11) = stats.se(2,1);  %S E Value
    
end

% Inflammation volume
for i = 1:numel(slices)
    [B,dev,stats] = mnrfit(inflammation_volumes(:,i),all_group_ids);
    
    csm_tstat_val(i,12) = stats.t(2,1);  % tstat Value
    csm_p_val(i,12) = stats.p(2,1);  % p Value
    csm_se_val(i,12) = stats.se(2,1);  %S E Value
    
end

