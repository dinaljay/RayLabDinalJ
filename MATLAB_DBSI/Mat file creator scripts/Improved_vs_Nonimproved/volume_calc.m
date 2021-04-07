% This script calculates the volume and surface area of the selected ROIs

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Improved_vs_Nonimproved/Improved';
out_dir_non_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Improved_vs_Nonimproved/Nonimproved';

%% Declare necessary variables

improv_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

non_improv_subjects = [5,6,9,11,12,13,14,20,22,25,27,30,34,37,41];

slices = (1:1:4);

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
        mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
        dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
        
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
        mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
        dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
        
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
