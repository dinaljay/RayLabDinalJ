% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROI_by_slices

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...
    "fiber1_extra_axial_map";"fiber1_extra_fraction_map";"fiber1_extra_radial_map";"fiber1_intra_axial_map";"fiber1_intra_fraction_map";...
    "fiber1_intra_radial_map";"fiber1_rgba_map_itk";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_extra_axial_map";...
    "fiber2_extra_fraction_map";"fiber2_extra_radial_map";"fiber2_intra_axial_map";"fiber2_intra_fraction_map";"fiber2_intra_radial_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

in_dir = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA';

slices = (1:1:4);

%% Create variable stores

for j = 1:numel(slices)
    
    slice_num = strcat('slice_',num2str(slices(j)));
    
    for n=1:numel(dhi_features)
        
        %Pre-op data
        file_name = strcat('control_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Pre_op/ROI_by_slice/Control/',file_name);
        load(temp);
        all_control_pre{n,1} = data_control(:,j);
        
        file_name = strcat('mild_csm_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Pre_op/ROI_by_slice/Mild_CSM/',file_name);
        load(temp);
        mild_csm_pre{n,1} = data_mild_csm(:,j);
        
        file_name = strcat('mod_csm_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Pre_op/ROI_by_slice/Moderate_CSM/',file_name);
        load(temp);
        mod_csm_pre{n,1} = data_mod_csm(:,j);
        
        clear data_control; clear data_mild_csm; clear data_mod_csm; clear temp;
        
        pre_controls = controls;
        pre_mild_cm_subjects = mild_cm_subjects;
        pre_moderate_cm_subjects = moderate_cm_subjects;
        
        %Post-op data
        file_name = strcat('control_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Post_op/ROI_by_slice/Control/',file_name);
        load(temp);
        all_control_post{n,1} = data_control(:,j);
        
        file_name = strcat('mild_csm_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Post_op/ROI_by_slice/Mild_CSM/',file_name);
        load(temp);
        mild_csm_post{n,1} = data_mild_csm(:,j);
        
        file_name = strcat('mod_csm_',dhi_features(n),'_data.mat');
        temp = fullfile(in_dir,'Post_op/ROI_by_slice/Moderate_CSM/',file_name);
        load(temp);
        mod_csm_post{n,1} = data_mod_csm(:,j);
        
        clear data_control; clear data_mild_csm; clear data_mod_csm; clear temp;
        
        post_controls = controls;
        post_mild_cm_subjects = mild_cm_subjects;
        post_moderate_cm_subjects = moderate_cm_subjects;
    end
    
    %% Create Pre-op patient IDs
    x=1;
    for k = 1:numel(pre_controls)
        
        pre_patientID{x,1} = strcat('CSM_C0',num2str(pre_controls(k)));
        x=x+1;
    end
    control_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    x=1;
    for k = 1:numel(pre_mild_cm_subjects)
        
        pre_patientID{x,1} = strcat('CSM_P0',num2str(pre_mild_cm_subjects(k)));
        x=x+1;
    end
    
    mild_csm_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    x=1;
    for k = 1:numel(pre_moderate_cm_subjects)
        
        pre_patientID{x,1} = strcat('CSM_P0',num2str(pre_moderate_cm_subjects(k)));
        x=x+1;
    end
    
    mod_csm_pre_patientID = repmat(pre_patientID, numel(dhi_features),1);
    clear pre_patientID;
    
    %Create post-patient IDs
    x=1;
    for k = 1:numel(post_controls)
        
        post_patientID{x,1} = strcat('CSM_C0',num2str(post_controls(k)));
        x=x+1;
    end
    control_post_patientID = repmat(post_patientID, numel(dhi_features),1);
    clear post_patientID;
    
    x=1;
    for k = 1:numel(post_mild_cm_subjects)
        
        post_patientID{x,1} = strcat('CSM_P0',num2str(post_mild_cm_subjects(k)));
        x=x+1;
    end
    mild_csm_post_patientID = repmat(post_patientID, numel(dhi_features),1);
    clear post_patientID;
    
    x=1;
    for k = 1:numel(post_moderate_cm_subjects)
        
        post_patientID{x,1} = strcat('CSM_P0',num2str(post_moderate_cm_subjects(k)));
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
    
    t_fin_sorted = table(t_fin.group, t_fin.PatientID, t_fin.operation, t_fin.feature_col, t_fin.data,...
            'VariableNames', {'Patient_Group', 'PatientID', 'Visit', 'MRI_Feature', 'Data_Value'});
    
    %% Save table
    
    out_dir = "/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice";
    terminal = strcat('all_ROI_by_',slice_num,'_data.csv');
    writetable(t_fin_sorted,fullfile(out_dir,terminal));
    
end

%% Create one giant csv file

t_slice_1 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_1_data.csv');
t_slice_2 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_2_data.csv');
t_slice_3 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_3_data.csv');
t_slice_4 = readtable('/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_ROI_by_slice/all_ROI_by_slice_4_data.csv');

slice_1 = repmat('Slice_1',height(t_slice_1),1);
slice_2 = repmat('Slice_2',height(t_slice_2),1);
slice_3 = repmat('Slice_3',height(t_slice_3),1);
slice_4 = repmat('Slice_4',height(t_slice_4),1);
slice_num = [slice_1;slice_2;slice_3;slice_4];

t_final = [t_slice_1;t_slice_2;t_slice_3;t_slice_4];
t_final.slice_num = slice_num;
t_final_sorted = table(t_final.slice_num,t_final.Patient_Group, t_final.PatientID, t_final.Visit, t_final.MRI_Feature, t_final.Data_Value, ...
    'VariableNames', {'Slice_Number', 'Patient_Group', 'PatientID', 'Visit', 'MRI_Feature', 'Data_Value'});


terminal = strcat('all_ROI_by_slice','_data.csv');
writetable(t_final_sorted,fullfile(out_dir,terminal));
