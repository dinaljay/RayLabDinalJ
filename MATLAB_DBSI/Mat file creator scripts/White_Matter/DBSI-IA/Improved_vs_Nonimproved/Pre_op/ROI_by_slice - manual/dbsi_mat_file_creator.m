% This script loads the ROI files for each slice for each patient and then
% extracts the relevant DBSI parameters and saves it as a mat file

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Improved_vs_Nonimproved/Pre_op/ROI_by_slice/Improved';
out_dir_non_improv = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Improved_vs_Nonimproved/Pre_op/ROI_by_slice/Nonimproved';

csm_roi = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Manual_ROIs/Patient';

%% Declare necessary variables

improv_subjects = [2,4,5,9,10,13,14,20,19,21,22,26,30,34,36,40,41,42,43,44,46,49]; %mJOA
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46]; %NASS

non_improv_subjects = [3,6,11,12,15,16,18,23,24,25,27,28,29,31,32,37,38,45,48,50];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF
% non_improv_subjects = [12,18,20,24,40]; %NASS

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...
    "fiber1_extra_axial_map";"fiber1_extra_fraction_map";"fiber1_extra_radial_map";"fiber1_intra_axial_map";"fiber1_intra_fraction_map";...
    "fiber1_intra_radial_map";"fiber1_rgba_map_itk";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_extra_axial_map";...
    "fiber2_extra_fraction_map";"fiber2_extra_radial_map";"fiber2_intra_axial_map";"fiber2_intra_fraction_map";"fiber2_intra_radial_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

%% Load patient data for each condition

fprintf('Improved Patients \n')

for i = 1:numel(dhi_features)
    data_improv = cell(numel(improv_subjects),1);
    
    for k = 1:numel(improv_subjects)
        
        subjectID = strcat('CSM_P0',num2str(improv_subjects(k)));
        disp(num2str(subjectID));
        temp =[];
        
        for j = 1:numel(slices)
            
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            file_name = strcat('JB_CSM_P_S',int2str(j),'roi_wm.nii.gz');
            mask_file = fullfile(csm_roi,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/',file_name);
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/dense/DHI_IA_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>=1);
            data_improv{k,j} = median(data, 'omitnan');
        end
        
    end
    terminal = strcat('improv_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_improv,terminal),'improv_subjects','data_improv');
    clear data_improv;
    fprintf('\n')
    
end

fprintf('\n')
fprintf('Nonimproved Patients \n')

for i = 1:numel(dhi_features)
    data_non_improv = cell(numel(non_improv_subjects),1);
    
    for k = 1:numel(non_improv_subjects)
        
        subjectID = strcat('CSM_P0',num2str(non_improv_subjects(k)));
        disp(num2str(subjectID));
        temp =[];
        
        for j = 1:numel(slices)
            
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            file_name = strcat('JB_CSM_P_S',int2str(j),'roi_wm.nii.gz');
            mask_file = fullfile(csm_roi,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/',file_name);
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/dense/DHI_IA_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>=1);
            data_non_improv{k,j} = median(data, 'omitnan');
        end
        
    end
    terminal = strcat('non_improv_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_non_improv,terminal),'non_improv_subjects','data_non_improv');
    clear data_non_improv;
    fprintf('\n')
    
end
