% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROI_by_slices

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["DTI ADC";"DTI Axial";"DTI FA";"DTI Radial";"Fiber Axial";"Fiber FA ";...
    "Fiber Radial";"Fiber Fraction";"Hindered Fraction";"Restricted Fraction";"Water Fraction";"Axon Volume";"Inflammation Volume"];
slices = (1:1:4);

for j = 1:numel(slices)
    
    slice_num = strcat('slice_',num2str(slices(j)));
    %% Create variable stores for Pre-op
    % %B0 Map
    %
    % load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI_by_slice/Pre_op/Control/control_b0_map_data.mat');
    % load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI_by_slice/Pre_op/CSM/csm_b0_map_data.mat');
    % b0 = [cell2mat(data_control);cell2mat(data_csm)];
    
    % DTI_ADC Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_dti_adc_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_dti_adc_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
    all_control_pre{1,1} = data_control(:,j);
    mild_csm_pre{1,1} = data_mild_csm(:,j);
    mod_csm_pre{1,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI Axial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_dti_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_dti_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
    all_control_pre{2,1} = data_control(:,j);
    mild_csm_pre{2,1} = data_mild_csm(:,j);
    mod_csm_pre{2,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI FA Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_dti_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_dti_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
    all_control_pre{3,1} = data_control(:,j);
    mild_csm_pre{3,1} = data_mild_csm(:,j);
    mod_csm_pre{3,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI Radial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_dti_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_dti_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
    all_control_pre{4,1} = data_control(:,j);
    mild_csm_pre{4,1} = data_mild_csm(:,j);
    mod_csm_pre{4,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Axial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_fiber1_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
    all_control_pre{5,1} = data_control(:,j);
    mild_csm_pre{5,1} = data_mild_csm(:,j);
    mod_csm_pre{5,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber FA Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_fiber1_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
    all_control_pre{6,1} = data_control(:,j);
    mild_csm_pre{6,1} = data_mild_csm(:,j);
    mod_csm_pre{6,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Radial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_fiber1_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
    all_control_pre{7,1} = data_control(:,j);
    mild_csm_pre{7,1} = data_mild_csm(:,j);
    mod_csm_pre{7,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_fiber_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
    all_control_pre{8,1} = data_control(:,j);
    mild_csm_pre{8,1} = data_mild_csm(:,j);
    mod_csm_pre{8,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Hindered Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_hindered_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
    all_control_pre{9,1} = data_control(:,j);
    mild_csm_pre{9,1} = data_mild_csm(:,j);
    mod_csm_pre{9,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Restricted Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_restricted_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
    all_control_pre{10,1} = data_control(:,j);
    mild_csm_pre{10,1} = data_mild_csm(:,j);
    mod_csm_pre{10,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Water Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_water_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_water_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
    all_control_pre{11,1} = data_control(:,j);
    mild_csm_pre{11,1} = data_mild_csm(:,j);
    mod_csm_pre{11,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    %Axon volume
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_axon_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_axon_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_axon_volume_data.mat');
    all_control_pre{12,1} = control_axon_volume(:,j);
    mild_csm_pre{12,1} = mild_csm_axon_volume(:,j);
    mod_csm_pre{12,1} = mod_csm_axon_volume(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    %Inflammation volume
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Control/control_inflammation_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Mild_CSM/mild_csm_inflammation_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_by_slice/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
    all_control_pre{13,1} = control_inflammation_volume(:,j);
    mild_csm_pre{13,1} = mild_csm_inflammation_volume(:,j);
    mod_csm_pre{13,1} = mod_csm_inflammation_volume(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    %% Create Pre-op patient IDs
    x=1;
    for k = 1:numel(controls)
        
        pre_patientID{x,1} = strcat('CSM_C0',num2str(controls(k)));
        x=x+1;
    end
    control_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    x=1;
    for k = 1:numel(mild_cm_subjects)
        
        pre_patientID{x,1} = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
        x=x+1;
    end
    
    mild_csm_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    x=1;
    for k = 1:numel(moderate_cm_subjects)
        
        pre_patientID{x,1} = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
        x=x+1;
    end
    
    mod_csm_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    
    %% Create variable stores for Post-op
    % %B0 Map
    %
    % load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI_by_slice/Post_op/Control/control_b0_map_data.mat');
    % load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/ROI_by_slice/Post_op/CSM/csm_b0_map_data.mat');
    % b0 = [cell2mat(data_control);cell2mat(data_csm)];
    
    % DTI_ADC Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_dti_adc_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_dti_adc_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
    all_control_post{1,1} = data_control(:,j);
    mild_csm_post{1,1} = data_mild_csm(:,j);
    mod_csm_post{1,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI Axial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_dti_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_dti_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
    all_control_post{2,1} = data_control(:,j);
    mild_csm_post{2,1} = data_mild_csm(:,j);
    mod_csm_post{2,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI FA Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_dti_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_dti_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
    all_control_post{3,1} = data_control(:,j);
    mild_csm_post{3,1} = data_mild_csm(:,j);
    mod_csm_post{3,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % DTI Radial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_dti_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_dti_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
    all_control_post{4,1} = data_control(:,j);
    mild_csm_post{4,1} = data_mild_csm(:,j);
    mod_csm_post{4,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Axial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_fiber1_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
    all_control_post{5,1} = data_control(:,j);
    mild_csm_post{5,1} = data_mild_csm(:,j);
    mod_csm_post{5,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber FA Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_fiber1_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
    all_control_post{6,1} = data_control(:,j);
    mild_csm_post{6,1} = data_mild_csm(:,j);
    mod_csm_post{6,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Radial Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_fiber1_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
    all_control_post{7,1} = data_control(:,j);
    mild_csm_post{7,1} = data_mild_csm(:,j);
    mod_csm_post{7,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Fiber Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_fiber_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
    all_control_post{8,1} = data_control(:,j);
    mild_csm_post{8,1} = data_mild_csm(:,j);
    mod_csm_post{8,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Hindered Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_hindered_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
    all_control_post{9,1} = data_control(:,j);
    mild_csm_post{9,1} = data_mild_csm(:,j);
    mod_csm_post{9,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Restricted Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_restricted_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
    all_control_post{10,1} = data_control(:,j);
    mild_csm_post{10,1} = data_mild_csm(:,j);
    mod_csm_post{10,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    % Water Fraction Map
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_water_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_water_fraction_map_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
    all_control_post{11,1} = data_control(:,j);
    mild_csm_post{11,1} = data_mild_csm(:,j);
    mod_csm_post{11,1} = data_mod_csm(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    %Axon volume
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_axon_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_axon_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_axon_volume_data.mat');
    all_control_post{12,1} = control_axon_volume(:,j);
    mild_csm_post{12,1} = mild_csm_axon_volume(:,j);
    mod_csm_post{12,1} = mod_csm_axon_volume(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    
    %Inflammation volume
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Control/control_inflammation_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Mild_CSM/mild_csm_inflammation_volume_data.mat');
    load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_by_slice/Moderate_CSM/mod_csm_inflammation_volume_data.mat');
    all_control_post{13,1} = control_inflammation_volume(:,j);
    mild_csm_post{13,1} = mild_csm_inflammation_volume(:,j);
    mod_csm_post{13,1} = mod_csm_inflammation_volume(:,j);
    
    clear data_control; clear data_mild_csm; clear data_mod_csm;
    
    %Create post-patient IDs
    x=1;
    for k = 1:numel(controls)
        
        post_patientID{x,1} = strcat('CSM_C0',num2str(controls(k)));
        x=x+1;
    end
    control_post_patientID = repmat(post_patientID, numel(dhi_features),1);
    clear post_patientID;
    
    x=1;
    for k = 1:numel(mild_cm_subjects)
        
        post_patientID{x,1} = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
        x=x+1;
    end
    mild_csm_post_patientID = repmat(post_patientID, numel(dhi_features),1);
    clear post_patientID;
    
    x=1;
    for k = 1:numel(moderate_cm_subjects)
        
        post_patientID{x,1} = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
        x=x+1;
    end
    
    mod_csm_post_patientID = repmat(post_patientID, numel(dhi_features),1);
    clear post_patientID;
    
    patientID = [control_pre_patientID; control_post_patientID;...
        mild_csm_pre_patientID; mild_csm_post_patientID;....
        mod_csm_pre_patientID; mod_csm_post_patientID];
    
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
    t_fin.PatientID = patientID;
    
    t_fin_sorted = table(t_fin.group, t_fin.PatientID, t_fin.operation, t_fin.feature_col, t_fin.data);
    
    %% Save table
    
    out_dir = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice";
    terminal = strcat('all_ROI_by_',slice_num,'_data.csv');
    writetable(t_fin_sorted,fullfile(out_dir,terminal));
    
end

%% Create one giant csv file

t_slice_1 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_1_data.csv');
t_slice_2 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_2_data.csv');
t_slice_3 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_3_data.csv');
t_slice_4 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DHI/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_4_data.csv');

slice_1 = repmat('Slice_1',height(t_slice_1),1);
slice_2 = repmat('Slice_2',height(t_slice_2),1);
slice_3 = repmat('Slice_3',height(t_slice_3),1);
slice_4 = repmat('Slice_4',height(t_slice_4),1);
slice_num = [slice_1;slice_2;slice_3;slice_4];

t_final = [t_slice_1;t_slice_2;t_slice_3;t_slice_4];
t_final.slice_num = slice_num;
t_final_sorted = table(t_final.slice_num,t_final.Var1, t_final.Var2, t_final.Var3, t_final.Var4, t_final.Var5, ...
	'VariableNames', {'Slice_Number', 'Patient_Group', 'PatientID', 'Visit', 'MRI_Feature', 'Data_Value'});


terminal = strcat('all_ROI_by_slice','_data.csv');
writetable(t_final_sorted,fullfile(out_dir,terminal));
