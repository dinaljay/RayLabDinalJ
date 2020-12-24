% This script does a rank sum test to see if there is a significant
% difference in mJOA scores between mild and moderate CSM patients

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% File paths

mjoa_path = '/home/functionalspinelab/Desktop/Dinal/DBSI_data/Clinical_data/csm_mjoa.csv';

%% Load data

scores = importdata(mjoa_path);

%% Declare necessary variables

controls = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20];

% awaiting confirmation on CSM subjects [1,11,]

% mild_cm_subjects = [1,2,3,4,10,15,16,17,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46,48,49,50];
mild_cm_subjects = [2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,36,38,40,42,43,44,45,46,48,49,50];

% moderate_cm_subjects = [5,6,7,8,9,11,12,13,14,20,22,25,27,30,34,35,37,39,41,47];
moderate_cm_subjects = [5,6,9,12,13,14,20,22,25,27,30,34,37,41];

cm_subjects = [mild_cm_subjects,moderate_cm_subjects];

%% Separate mjoa scores into moderate and mild
n=1;
for i =1:length(mild_cm_subjects)
    for j = 1:length(scores)
        
        if scores(j,1) == mild_cm_subjects(1,i)
            mild_mjoa(n,1)= scores(j,2);
            n=n+1;
        end
    end
end

n=1;
for i =1:length(moderate_cm_subjects)
    for j = 1:length(scores)
        
        if scores(j,1) == moderate_cm_subjects(1,i)
            moderate_mjoa(n,1)= scores(j,2);
            n=n+1;
        end
    end
end

%% ranksum test

[p_mjoa,~,stats_array_mjoa] = ranksum(mild_mjoa,moderate_mjoa,'method','approximate');
zval_mjoa = stats_array_mjoa.zval;

