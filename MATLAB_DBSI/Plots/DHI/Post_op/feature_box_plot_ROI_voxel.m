% This script plots boxplots to show the distribution of variables for each
% DBSI feature based on ROIs

close all;
clear all;
addpath (genpath('/home/functionalspinelab/Documents/MATLAB'));
addpath (genpath('/home/functionalspinelab/Desktop/Dinal/Scripts/MATLAB_DBSI'));

%% Declare necessary variables

dhi_features = ["DTI ADC";"DTI Axial";"DTI FA";"DTI Radial";"Fiber Axial";"Fiber FA ";...
    "Fiber Radial";"Fiber Fraction";"Hindered Fraction";"Restricted Fraction";"Water Fraction"];

%% Create variable stores
% %B0 Map
%
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Control/control_b0_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/CSM/csm_b0_map_data.mat');
% b0 = [cell2mat(data_control);cell2mat(data_csm)];
% clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI_ADC Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_adc_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_adc_map_data.mat');
all_control{1,1} = data_control;
mild_csm{1,1} = data_mild_csm;
mod_csm{1,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_axial_map_data.mat');
all_control{2,1} = data_control;
mild_csm{2,1} = data_mild_csm;
mod_csm{2,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_fa_map_data.mat');
all_control{3,1} = data_control;
mild_csm{3,1} = data_mild_csm;
mod_csm{3,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% DTI Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_dti_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_dti_radial_map_data.mat');
all_control{4,1} = data_control;
mild_csm{4,1} = data_mild_csm;
mod_csm{4,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Axial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_axial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_axial_map_data.mat');
all_control{5,1} = data_control;
mild_csm{5,1} = data_mild_csm;
mod_csm{5,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber FA Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_fa_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_fa_map_data.mat');
all_control{6,1} = data_control;
mild_csm{6,1} = data_mild_csm;
mod_csm{6,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Radial Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber1_radial_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber1_radial_map_data.mat');
all_control{7,1} = data_control;
mild_csm{7,1} = data_mild_csm;
mod_csm{7,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Fiber Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_fiber_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_fiber_fraction_map_data.mat');
all_control{8,1} = data_control;
mild_csm{8,1} = data_mild_csm;
mod_csm{8,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Hindered Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_hindered_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_hindered_fraction_map_data.mat');
all_control{9,1} = data_control;
mild_csm{9,1} = data_mild_csm;
mod_csm{9,1} = data_mod_csm;;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Restricted Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_restricted_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_restricted_fraction_map_data.mat');
all_control{10,1} = data_control;
mild_csm{10,1} = data_mild_csm;
mod_csm{10,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% Water Fraction Map
% Water Fraction Map
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_water_fraction_map_data.mat');
load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_water_fraction_map_data.mat');
all_control{11,1} = data_control;
mild_csm{11,1} = data_mild_csm;
mod_csm{11,1} = data_mod_csm;
clear data_control; clear data_mild_csm; clear data_mod_csm;

% %Axon volume
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_axon_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_axon_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_axon_volume_map_data.mat');
% all_control{12,1} = data_control;
% mild_csm{12,1} = data_mild_csm;
% mod_csm{12,1} = data_mod_csm;

% %Inflammation volume
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Control/control_inflammation_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Mild_CSM/mild_csm_inflammation_volume_map_data.mat');
% load('/media/functionalspinelab/RAID/Data/Dinal/MATLAB_Data/DBSI/White_Matter/DHI/Post_op/ROI_Voxel/All_slices/Moderate_CSM/mod_csm_inflammation_volume_map_data.mat');
% all_control{13,1} = data_control;
% mild_csm{13,1} = data_mild_csm;
% mod_csm{13,1} = data_mod_csm;

%% Plots
colors = [1, 0, 0; 0, 1, 0; 0, 0, 1];

for i = 1:numel(dhi_features)
    figure
    
    x1 = cell2mat(all_control{i,1});
    x2 = cell2mat(mild_csm{i,1});
    x3 = cell2mat(mod_csm{i,1});
    x = [x1; x2; x3];
    
    g1 = repmat({'Control'},length(x1),1);
    g2 = repmat({'Mild'},length(x2),1);
    g3 = repmat({'Mod'},length(x3),1);
    g = [g1; g2; g3];
    
    %subplot(6,2,i);
    boxplot(x,g,'Notch', 'on','Labels',{'Controls','Mild CSM', 'Moderate CSM'},'Whisker',0.5)
    temp = strcat(dhi_features(i),"");
    title(sprintf('%s',temp))
    
    h = findobj(gca,'Tag','Box');
    for j=1:length(h)
        patch(get(h(j),'XData'),get(h(j),'YData'),colors(j,:),'FaceAlpha',.5);
    end
    
end


