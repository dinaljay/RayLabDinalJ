% This script calculates the volume and surface area of the selected ROIs

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Improved';
out_dir_non_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Improved_vs_Nonimproved/Pre_op/ROI/Nonimproved';

csm_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Patient';

%% Declare necessary variables

improv_subjects = [2,4,5,9,10,13,14,20,19,21,22,26,30,34,36,40,41,42,43,44,46,49]; %mJOA
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46]; %NASS

non_improv_subjects = [3,6,11,12,15,16,18,23,24,25,27,28,29,31,32,37,38,45,48,50];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF
% non_improv_subjects = [12,18,20,24,40]; %NASS

slices = (1:1:4);
% slices = [4];

voxel_volume = 0.35*0.35*7.5;

voxel_surface_area = 0.35*0.35;

%% Load patient data for each condition

fprintf('Improved Patients \n')

data_improv = cell(numel(improv_subjects),1);

for k = 1:numel(improv_subjects)
    
    subjectID = strcat('CSM_P0',num2str(improv_subjects(k)));
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
        data = dwi_data(expert_rois>0.7);
        temp = [temp;data];
    end
    data_improv{k,1} = length(temp);
end

fprintf('\n')
fprintf('Nonimproved Patients \n')

data_non_improv = cell(numel(non_improv_subjects),1);

for k = 1:numel(non_improv_subjects)
    
    subjectID = strcat('CSM_P0',num2str(non_improv_subjects(k)));
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
        data = dwi_data(expert_rois>0.7);
        temp = [temp;data];
    end
    data_non_improv{k,1} = length(temp);
end

%% Caculate volume and surface area

% Improved subjects
for i = 1:length(data_improv)
    improv_volumes{i,1} = data_improv{i,1}*voxel_volume;
    improv_surface_area{i,1} = data_improv{i,1}*voxel_surface_area;
end

terminal = strcat('improv_volume','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_volumes');

terminal = strcat('improv_surface_area','_data.mat');
save(fullfile(out_dir_improv,terminal),'improv_subjects','improv_surface_area');

% Moderate CM subjects
for i = 1:length(data_non_improv)
    non_improv_volumes{i,1} = data_non_improv{i,1}*voxel_volume;
    non_improv_surface_area{i,1} = data_non_improv{i,1}*voxel_surface_area;
end

terminal = strcat('non_improv_volume','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_volumes');

terminal = strcat('non_improv_surface_area','_data.mat');
save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','non_improv_surface_area');
