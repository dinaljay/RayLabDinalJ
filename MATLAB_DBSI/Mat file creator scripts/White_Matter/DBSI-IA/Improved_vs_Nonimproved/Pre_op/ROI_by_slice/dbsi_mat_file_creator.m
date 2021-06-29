% This script loads the ROI files for each slice for each patient and then
% extracts the relevant DBSI parameters and saves it as a mat file

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_control = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Pre_op/ROI_by_slice/Control';

out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Pre_op/ROI_by_slice/Mild_CSM';
out_dir_mod_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Pre_op/ROI_by_slice/Moderate_CSM';


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

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"dti_radial_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load patient data for each condition

fprintf('Control Patients \n')

for i = 1:numel(dhi_features)
    data_control = cell(numel(controls),1);
    disp(dhi_features(i))
    
    for k = 1:numel(controls)
        subjectID = strcat('CSM_C0',num2str(controls(k)));
        disp(num2str(subjectID));
        temp = [];
        for j = 1:numel(slices)
            
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
            dwi_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            
            data = dwi_data(expert_rois>0.7);
            data_control{k,j} = median(data);
        end
    end
    
    terminal = strcat('control_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_control,terminal),'controls','data_control');
    clear data_control;
    fprintf('\n')
    
end

fprintf('\n')
fprintf('Mild Cervical Myelopathy Patients \n')

for i = 1:numel(dhi_features)
    data_mild_csm = cell(numel(mild_cm_subjects),1);
    
    for k = 1:numel(mild_cm_subjects)
        
        subjectID = strcat('CSM_P0',num2str(mild_cm_subjects(k)));
        disp(num2str(subjectID));
        temp =[];
        
        for j = 1:numel(slices)
            
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>0.7);
            data_mild_csm{k,j} = median(data);
        end
        
    end
    terminal = strcat('mild_csm_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_mild_csm,terminal),'mild_cm_subjects','data_mild_csm');
    clear data_mild_csm;
    fprintf('\n')
    
end

fprintf('\n')
fprintf('Moderate Cervical Myelopathy Patients \n')

for i = 1:numel(dhi_features)
    data_mod_csm = cell(numel(moderate_cm_subjects),1);
    
    for k = 1:numel(moderate_cm_subjects)
        
        subjectID = strcat('CSM_P0',num2str(moderate_cm_subjects(k)));
        disp(num2str(subjectID));
        temp =[];
        
        for j = 1:numel(slices)
            
            slice_num = strcat('slice_',num2str(slices(j)));
            disp(num2str(slice_num));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>0.7);
            data_mod_csm{k,j} = median(data);
        end
        
    end
    terminal = strcat('mod_csm_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_mod_csm,terminal),'moderate_cm_subjects','data_mod_csm');
    clear data_mod_csm;
    fprintf('\n')
    
end
