% This script loads the ROI files for each slice for each patient and then
% extracts the relevant DBSI parameters and saves it as a mat file

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM_grouped';

% out_dir_mild_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Mild_CSM';
% out_dir_moderate_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/Moderate_CSM';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% awaiting confirmation on CSM subjects [1,11,]

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load patient data for controls ordered by CSM

fprintf('All Cervical Myelopathy Patients \n')

for i = 1:numel(dhi_features)
    data_csm = cell(numel(cm_subjects),numel(slices));
    
    for j = 1:numel(slices)
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        for k = 1:numel(cm_subjects)
            
            subjectID = strcat('CSM_P0',num2str(cm_subjects(k)));
            disp(num2str(subjectID));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            data = dwi_data(expert_rois>0.7);
            data_csm{k,j} = mean(data);
        end
        fprintf('\n')
    end
    terminal = strcat('csm_',dhi_features(i),'_data.mat');
    save(fullfile(out_dir_csm,terminal),'cm_subjects','mild_cm_subjects','moderate_cm_subjects','data_csm');
    fprintf('\n')
    
end

return;

