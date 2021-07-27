% This script calculates the volume and surface area of the selected ROIs

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Control';
out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Mild_CSM';
out_dir_mod_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Pre_op/ROI/Moderate_CSM';

csm_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Patient';
control_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Control';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];
%CSM_P01 template no good

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);
% slices = [4];

voxel_volume = 0.35*0.35*7.5;

voxel_surface_area = 0.35*0.35;

%% Load patient data for each condition

fprintf('Control Patients \n')

data_control = cell(numel(controls),1);

for k = 1:numel(controls)
    
    subjectID = strcat('CSM_C0',num2str(controls(k)));
    disp(num2str(subjectID));
    temp = [];
    
    for j = 1:numel(slices)
        
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        param_file =('dti_fa_map.nii');
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
    data_control{k,1} = length(temp);
end

fprintf('\n')
fprintf('Mild Cervical Myelopathy Patients \n')

data_mild_csm = cell(numel(mild_cm_subjects),1);

for k = 1:numel(mild_cm_subjects)
    
    subjectID = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
    disp(num2str(subjectID));
    temp = [];
    
    for j = 1:numel(slices)
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        param_file =('dti_fa_map.nii');
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
    data_mild_csm{k,1} = length(temp);
end

fprintf('\n')
fprintf('Moderate Cervical Myelopathy Patients \n')

data_mod_csm = cell(numel(moderate_cm_subjects),1);

for k = 1:numel(moderate_cm_subjects)
    
    subjectID = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
    disp(num2str(subjectID));
    temp = [];
    
    for j = 1:numel(slices)
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        param_file =('dti_fa_map.nii');
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
    data_mod_csm{k,1} = length(temp);
end

%% Caculate volume and surface area

% Controls
for i = 1:length(data_control)
    control_volumes{i,1} = data_control{i,1}*voxel_volume;
    control_surface_area{i,1} = data_control{i,1}*voxel_surface_area;
end

terminal = strcat('control_volume','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_volumes');

terminal = strcat('control_surface_area','_data.mat');
save(fullfile(out_dir_control,terminal),'controls','control_surface_area');

% Mild CM subjects
for i = 1:length(data_mild_csm)
    mild_csm_volumes{i,1} = data_mild_csm{i,1}*voxel_volume;
    mild_csm_surface_area{i,1} = data_mild_csm{i,1}*voxel_surface_area;
end

terminal = strcat('mild_csm_volume','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_volumes');

terminal = strcat('mild_csm_surface_area','_data.mat');
save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','mild_csm_surface_area');

% Moderate CM subjects
for i = 1:length(data_mod_csm)
    mod_csm_volumes{i,1} = data_mod_csm{i,1}*voxel_volume;
    mod_csm_surface_area{i,1} = data_mod_csm{i,1}*voxel_surface_area;
end

terminal = strcat('mod_csm_volume','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_volumes');

terminal = strcat('mod_csm_surface_area','_data.mat');
save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','mod_csm_surface_area');
