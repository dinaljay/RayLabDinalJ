% This script creates a .csv file of a patient's age

clear all;
close all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
out_dir = '/media/functionalspinelab/RAID/Data/Dinal/Pycharm_Data/DBSI_CSV_Data';
ages_path = '';

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41,47];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

cm_subjects = sort(cm_subjects,2);

ages = importdata(ages_path);
ages_data = ages.data;

%% Get necessary ages only

for i = 1:length(ages_data)
    n=1;
    for k = 1:numel(controls)
        ptID = strcat('Control_',controls(i));
        if (ages_data(i,1) == ptID)
            all_ages(n,1) = ptID;
            all_ages(n,2) = ages_data(i,2);
            n=n+1;
        end
    end
    
    for k = 1:numel(cm_subjects)
        ptID = strcat('CSM_',cm_subjects(i));
        if (ages_data(i,1) == ptID)
            all_ages(n,1) = ptID;
            all_ages(n,2) = ages_data(i,2);
            n=n+1;
        end
    end
    
end

%% Generate .csv file

group_id = categorical([repmat({'Control'},numel(controls),1);repmat({'CSM'},numel(cm_subjects),1)]);

terminal2 = strcat('all_ages','_data.csv');
table_out=table(group_id,all_ages);
table_out.Properties.VariableNames{1} = 'Group';
table_out.Properties.VariableNames{2} = 'Age';
writetable(table_out,fullfile(out_dir_all,terminal2));

