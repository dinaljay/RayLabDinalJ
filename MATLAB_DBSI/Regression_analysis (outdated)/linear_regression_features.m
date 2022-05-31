% This script does a linear regression using each DBSI feature to predict
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

load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_b0_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_b0_map_data.mat');
b0 = [cell2mat(data_control);cell2mat(data_csm)];

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_adc_map_data.mat');
dti_adc = [cell2mat(data_control);cell2mat(data_csm)];

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_axial_map_data.mat');
dti_axial = [cell2mat(data_control);cell2mat(data_csm)];

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_dti_fa_map_data.mat');
dti_fa = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_axial_map_data.mat');
fiber_axial = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_fa_map_data.mat');
fiber_fa = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber1_radial_map_data.mat');
fiber_radial = [cell2mat(data_control);cell2mat(data_csm)];

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_fiber_fraction_map_data.mat');
fiber_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_hindered_fraction_map_data.mat');
hindered_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_restricted_fraction_map_data.mat');
restricted_fraction = [cell2mat(data_control);cell2mat(data_csm)];

% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/CSM/csm_water_fraction_map_data.mat');
water_fraction = [cell2mat(data_control);cell2mat(data_csm)];

%% Linear regression

control_group = zeros(numel(controls),1);
csm_group = ones(numel(cm_subjects),1);
all_group_ids = [control_group;csm_group];

for i = 1:numel(slices)
    
    tbl=table(all_group_ids,dti_adc(:,i),dti_axial(:,i),dti_fa(:,i),fiber_axial(:,i),fiber_fa(:,i),fiber_radial(:,i),...
        fiber_fraction(:,i),hindered_fraction(:,i),restricted_fraction(:,i),water_fraction(:,i),'VariableNames',{'Group','DTI_ADC','DTI_Axial',...
        'DTI_FA','Fiber_Axial','Fiber_FA','Fiber_Radial','Fiber_Fraction','Hindered_Fraction','Restricted_Fraction','Water_Fraction'});
    
    X=fitlm(tbl,'Group~DTI_ADC+DTI_Axial+DTI_FA+Fiber_Axial+Fiber_FA+Fiber_Radial+Fiber_Fraction+Hindered_Fraction+Restricted_Fraction+Water_Fraction');

    for j = 1:(size(X.Coefficients)-1)
        csm_b_val(i,j) = X.Coefficients.Estimate(j+1);  % B Value
        csm_p_val(i,j) = X.Coefficients.pValue(j+1);  % p Value
        csm_se_val(i,j) = X.Coefficients.SE(j+1);  %S E Value
    end
    
end
