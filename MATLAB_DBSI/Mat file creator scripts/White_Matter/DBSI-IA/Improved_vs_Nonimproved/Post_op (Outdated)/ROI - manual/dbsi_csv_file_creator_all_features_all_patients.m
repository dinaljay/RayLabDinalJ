% This script creates a .csv file for all patients including all features

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

improv_subjects = [2,4,5,9,10,13,14,20,19,21,22,26,30,34,36,40,41,42,43,44,46,49]; %mJOA
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46];  %SF-36 PF
% improv_subjects = [2,3,5,9,15,19,21,23,26,27,28,29,30,36,41,45,46]; %NASS

non_improv_subjects = [3,6,11,12,15,16,18,23,24,25,27,28,29,31,32,37,38,45,48,50];  %mJOA
% non_improv_subjects = [12,18,20,24,40]; %SF-36 PF
% non_improv_subjects = [12,18,20,24,40]; %NASS

% slices = (1:1:4);
slices = [4];

dhi_features = ["b0_map";"dti_adc_map";"dti_axial_map";"dti_b_map";"dti_dirx_map";"dti_diry_map";"dti_fa_map";"dti_g_map";...
    "dti_radial_map";"dti_rgba_map";"dti_rgba_map_itk";"dti_r_map";"fiber1_dirx_map";"fiber1_diry_map";"fiber1_dirz_map";...
    "fiber1_extra_axial_map";"fiber1_extra_fraction_map";"fiber1_extra_radial_map";"fiber1_intra_axial_map";"fiber1_intra_fraction_map";...
    "fiber1_intra_radial_map";"fiber1_rgba_map_itk";"fiber2_dirx_map";"fiber2_diry_map";"fiber2_dirz_map";"fiber2_extra_axial_map";...
    "fiber2_extra_fraction_map";"fiber2_extra_radial_map";"fiber2_intra_axial_map";"fiber2_intra_fraction_map";"fiber2_intra_radial_map";...
    "fraction_rgba_map";"hindered_adc_map";"hindered_fraction_map";"iso_adc_map";"model_v_map";"restricted_adc_map";"restricted_fraction_map";...
    "water_adc_map";"water_fraction_map"];

in_dir = '/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DBSI-IA/Improved_vs_Nonimproved';

%% Create variable store

for n=1:numel(dhi_features)
    
    file_name = strcat('improv_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Post_op/ROI/Improved/',file_name);
    load(temp);
    improv_pre{n,1} = data_improv;
    
    file_name = strcat('non_improv_',dhi_features(n),'_data.mat');
    temp = fullfile(in_dir,'Post_op/ROI/Nonimproved/',file_name);
    load(temp);
    non_improv_pre{n,1} = data_non_improv;
    
    clear data_improv; clear data_non_improv; clear temp;
    
end

%% Create patient IDs
x=1;

for k = 1:numel(improv_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(improv_subjects(k)));
    x=x+1;
end

for k = 1:numel(non_improv_subjects)
    
    patientID{x,1} = strcat('CSM_P0',num2str(non_improv_subjects(k)));
    x=x+1;
end

%% Generate csv file

var_Names = ["Patient_ID", "Group", "Group_ID", "b0_map", "dti_adc_map", "dti_axial_map", "dti_b_map", "dti_dirx_map", "dti_diry_map", "dti_fa_map", "dti_g_map",...
                    "dti_radial_map", "dti_rgba_map", "dti_rgba_map_itk", "dti_r_map", "fiber1_dirx_map", "fiber1_diry_map", "fiber1_dirz_map",...
                    "fiber1_extra_axial_map", "fiber1_extra_fraction_map", "fiber1_extra_radial_map", "fiber1_intra_axial_map", "fiber1_intra_fraction_map",...
                    "fiber1_intra_radial_map", "fiber1_rgba_map_itk", "fiber2_dirx_map", "fiber2_diry_map", "fiber2_dirz_map", "fiber2_extra_axial_map",...
                    "fiber2_extra_fraction_map", "fiber2_extra_radial_map", "fiber2_intra_axial_map", "fiber2_intra_fraction_map", "fiber2_intra_radial_map",...
                    "fraction_rgba_map", "hindered_adc_map", "hindered_fraction_map", "iso_adc_map", "model_v_map", "restricted_adc_map", "restricted_fraction_map",...
                    "water_adc_map", "water_fraction_map"];
            
out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/White_Matter/DBSI-IA/Pycharm_Data_improv_vs_nonimprov/Post_op/ROI';


%Improv vs Non-improv 
group = categorical([repmat({'Improved'},numel(improv_subjects),1);repmat({'Non-Improved'},numel(non_improv_subjects),1)]);
group_id = categorical([repmat({'0'},numel(improv_subjects),1);repmat({'1'},numel(non_improv_subjects),1)]);
data = [];

for i=1:length(dhi_features)
    
    data1(:,i) = cell2mat(improv_pre{i,1});
    
end

for i=1:length(dhi_features)
    
    data2(:,i) = cell2mat(non_improv_pre{i,1});
    
end

data_fin = [data1;data2];

t1 = table(patientID, group, group_id);
t2 = array2table(data_fin);
table_out = [t1,t2];
table_out = renamevars(table_out,1:width(table_out),var_Names);

terminal2 = strcat('all_patients_all_features','_data.csv');
writetable(table_out,fullfile(out_dir,terminal2));

