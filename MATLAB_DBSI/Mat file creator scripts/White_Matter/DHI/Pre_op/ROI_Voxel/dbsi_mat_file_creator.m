% This script loads the ROI files for each slice for each patient and then
% extracts the relevant DBSI parameters and saves all the voxel parameters
% for that ROI as a mat file

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';

out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Control';
out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Mild_CSM';
out_dir_mod_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI_Voxel/All_slices/Moderate_CSM';

csm_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Patient';
control_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Control';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
% mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50]; %Dinal's classification
%CSM_P01 template no good
mild_cm_subjects = [3,4,15,16,18,19,21,23,24,28,29,32,36,38,40,42,43,44,45,46,48,49,50]; %Justin's classification

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
% moderate_cm_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41]; %Dinal's classification

moderate_cm_subjects = [2,5,6,9,10,11,12,13,14,20,22,25,26,27,30,31,34,37,41]; %Justin's classification

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_dirz_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_axial_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...    
    "fiber1_fa_map";"fiber1_fiber_fraction_map";"fiber1_radial_map";"fiber1_rgba_map";"fiber1_rgba_map_itk";...
    "fiber2_axial_map";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_fa_map";...
    "fiber2_fiber_fraction_map";"fiber2_radial_map";"fiber_fraction_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

%% Load control data

for i = 1:numel(dhi_features)
    data_control = {};
    disp(dhi_features(i))
    new_temp = [];
    
    for k = 1:numel(controls)
        
        subjectID = strcat('CSM_C0',num2str(controls(k)));
        disp(num2str(subjectID));
        
        for j = 1:numel(slices)
            temp = [];
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            file_name = strcat('JB_CSM_C_S',int2str(j),'roi_wm.nii.gz');
            mask_file = fullfile(control_roi,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/',file_name);
            dwi_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/dense/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            
            data = dwi_data(expert_rois>=1);
            temp = [temp;data];
        end
        new_temp = [new_temp;temp];
        fprintf('\n')
    end
    
    data_control = num2cell(new_temp);
    terminal = strcat('control_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_control,terminal),'controls','data_control');
    clear data_control;
    
end

%% Load patient data for Mild CSM

fprintf('Mild Cervical Myelopathy Patients \n')

for i = 1:numel(dhi_features)
    data_mild_csm = {};
    new_temp = [];
    
    for k = 1:numel(mild_cm_subjects)
        
        subjectID = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
        disp(num2str(subjectID));
        
        for j = 1:numel(slices)
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            temp = [];
            
            param_file = strcat(dhi_features(i),'.nii');
            file_name = strcat('JB_CSM_P_S',int2str(j),'roi_wm.nii.gz');
            mask_file = fullfile(csm_roi,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/',file_name);
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/dense/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>=1);
            temp = [temp;data];
        end
        new_temp = [new_temp;temp];
        fprintf('\n')
    end
    data_mild_csm = num2cell(new_temp);
    terminal = strcat('mild_csm_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_mild_csm,terminal),'cm_subjects','mild_cm_subjects','data_mild_csm');
    clear data_mild_csm;
    
end

%% Load patient data for Moderate CSM

fprintf('Moderate Cervical Myelopathy Patients \n')

for i = 1:numel(dhi_features)
    data_mod_csm = {};
    new_temp = [];
    
    for k = 1:numel(moderate_cm_subjects)
        
        subjectID = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
        disp(num2str(subjectID));
        
        for j = 1:numel(slices)
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            temp = [];
            
            param_file = strcat(dhi_features(i),'.nii');
            file_name = strcat('JB_CSM_P_S',int2str(j),'roi_wm.nii.gz');
            mask_file = fullfile(csm_roi,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/',file_name);
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/dense/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>=1);
            temp = [temp;data];
        end
        new_temp = [new_temp;temp];
        fprintf('\n')
    end
    data_mod_csm = num2cell(new_temp);
    terminal = strcat('mod_csm_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_mod_csm,terminal),'cm_subjects','moderate_cm_subjects','data_mod_csm');
    clear data_mod_csm;
    
end


