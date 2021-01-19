% This script loads the probability files for each slice for each patient
% and then creates a NIfTI version of the ROI file for viewing purposes

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_csm = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/ROI/CSM';

%% Declare necessary variables

% controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];
%
% % awaiting confirmation on CSM subjects [1,11,]
%
% % mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
% mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];
%
% % moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
% moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41];

mild_cm_subjects = [2,3,4,10,15];

moderate_cm_subjects = [20,22,25,27,30];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

slices = (1:1:4);

%% Load patient data for each condition

% slice_num = 'slice_2';

% Controls

% for k = 1:numel(controls)
%     
%     subjectID = strcat('CSM_C0',num2str(controls(k)));
%     disp(num2str(subjectID));
%     
%     for j = 1:numel(slices)
%         
%         slice_num = strcat('slice_',num2str(slices(j)));
%         
%         mask_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
%         
%         mask = niftiread(mask_file);
%         
%         expert_rois = double(mask);
%         expert_rois(expert_rois>0.7) = 1;
%         expert_rois(expert_rois<0.7) = 0;
%         
%         terminal = 'roi_voxel';
%         save_path = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit',slice_num,'all_volumes/DHI_results_0.3_0.3_3_3',terminal);
%         
%         niftiwrite(expert_rois, save_path);
%         clear expert_rois
%     end
%     
% end

% All CM subjects

for k = 1:numel(cm_subjects)
    
    subjectID = strcat('CSM_P0',num2str(cm_subjects(k)));
    disp(num2str(subjectID));
    
    for j = 1:numel(slices)
        
        slice_num = strcat('slice_',num2str(slices(j)));
        
        mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit',slice_num,'/all_volumes/label/template/PAM50_wm.nii.gz');
        
        mask = niftiread(mask_file);
        
        expert_rois = double(mask);
        expert_rois(expert_rois>0.7) = 1;
        expert_rois(expert_rois<0.7) = 0;
        
        terminal = 'roi_voxel';
        save_path = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit',slice_num,'all_volumes/DHI_results_0.3_0.3_3_3',terminal);
        
        niftiwrite(expert_rois, save_path);
        clear expert_rois
    end
    
end



