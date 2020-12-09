% This script loads the ROI files for each slice for each patient and then
% extracts the relevant DBSI parameters and saves it as a csv file for all
% patients

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir_all = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data_Voxel/DBSI_CSV_Data/All_patients';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41,47];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

slices = (1:1:4);

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_fa_map";"fiber1_axial_map";"fiber1_fa_map";...
    "fiber1_radial_map";"fiber_fraction_map";"hindered_fraction_map";"restricted_fraction_map";"water_fraction_map"];

%% Load patient data for each condition

fprintf('Control Patients \n')

group_id = categorical([repmat({'Control'},numel(controls),1);repmat({'CSM'},numel(cm_subjects),1)]);

for i = 1:numel(dhi_features)
    data_control = cell(numel(controls),numel(slices));
    disp(dhi_features(i))
    
    for j = 1:numel(slices)
        
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        for k = 1:numel(controls)
            
            subjectID = strcat('CSM_C0',num2str(controls(k)));
            disp(num2str(subjectID));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/roi.nii.gz');
            dwi_file = fullfile(control_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            
            data = dwi_data(expert_rois>0);
            data_control{k,j} = mean(data);
        end
        
    end
    
    data_csm = cell(numel(cm_subjects),numel(slices));
    
    for j = 1:numel(slices)
        slice_num = strcat('slice_',num2str(slices(j)));
        disp(num2str(slice_num));
        
        for k = 1:numel(cm_subjects)
            
            subjectID = strcat('CSM_P0',num2str(cm_subjects(k)));
            disp(num2str(subjectID));
            
            param_file = strcat(dhi_features(i),'.nii');
            mask_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3/roi.nii.gz');
            dwi_file = fullfile(csm_path,subjectID,'/scan_1/dMRI_ZOOMit/',slice_num,'/all_volumes/DHI_results_0.3_0.3_3_3',param_file);
            
            mask = niftiread(mask_file);
            dwi_data = niftiread(dwi_file);
            
            expert_rois = double(mask);
            dwi_data = double(dwi_data);
            
            data_csm{k,j} = mean(dwi_data(expert_rois>0));
        end
        
    end
    
    all_data = cell2mat([data_control;data_csm]);
    
    all_data_1 = all_data(:,1);
    all_data_2 = all_data(:,2);
    all_data_3 = all_data(:,3);
    all_data_4 = all_data(:,4);
    
    
    terminal2 = strcat('all_',dhi_features(i),'_data.csv');
    table_out=table(group_id,all_data_1,all_data_2,all_data_3,all_data_4);
    table_out.Properties.VariableNames{1} = 'Group';
    table_out.Properties.VariableNames{2} = 'Slice_1';
    table_out.Properties.VariableNames{3} = 'Slice_2';
    table_out.Properties.VariableNames{4} = 'Slice_3';
    table_out.Properties.VariableNames{5} = 'Slice_4';
    
    writetable(table_out,fullfile(out_dir_all,terminal2));
    fprintf('\n')
    
end
