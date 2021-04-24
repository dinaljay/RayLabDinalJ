import matlab.engine
import os
import sys
import configparser

#Patient IDs

controls = [4,5,8,9,10,11,15,16,17,18]
csm_patients = [2,3,5,15,18,19,21,23,24,26,28,29,30,36,40,41,45,46,9,12,17,20,27]

#Other information
control_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Control';
csm_path = '/media/functionalspinelab/RAID/Data/Dinal/DBSI_Data/CSM_New/Patient';
slices = [1,2,3,4]

first_time = 'True' #Acceptable answers include True or False. If this is the first time running this code on this machine, the code will create the config file input MAT file needed for the DBSI calculations.

# define functions

def dhi_config(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    Config = configparser.ConfigParser()
    cfgfile = open("config.ini",'w')
    Config.add_section('INPUT')
    Config.set('INPUT','data_dir',current_dir)
    path,dwi_file = os.path.split(nii_file)
    path,bval = os.path.split(os.path.join(current_dir + os.sep + 'bval'))
    path,bvec = os.path.split(os.path.join(current_dir + os.sep + 'bvec'))
    Config.set('INPUT','dwi_file',"%s"%dwi_file)
    Config.set('INPUT','mask_file',"NA")
    Config.set('INPUT','rotation_matrix','NA')
    Config.set('INPUT','bval_file',"%s"%bval)
    Config.set('INPUT','bvec_file',"%s"%bvec)
    Config.set('INPUT','preprocess','NA')
    Config.set('INPUT','slices_to_compute','0')
    Config.set('INPUT','dhi_mode','map')
    Config.set('INPUT','norm_by_bvec','no')
    Config.set('INPUT','bmax_dhi',' ')
    Config.set('INPUT','dhi_input_file','dhi_input.mat')
    Config.add_section('DHI')
    Config.set('DHI','dhi_input_file','dhi_input.mat')
    Config.set('DHI','dhi_config_file','/home/functionalspinelab/Documents/MATLAB/dhi_release/Configuration_DHI_IA_Human.mat')
    Config.set('DHI','dhi_class_file','dhi_class.mat')
    Config.add_section('OUTPUT')
    Config.set('OUTPUT','output_option','2')
    Config.set('OUTPUT','output_format','nii')
    Config.set('OUTPUT','iso_threshold','0.3,0.3,3.0,3.0')
    Config.set('OUTPUT','output_fib','0')
    Config.set('OUTPUT','output_fib_res','1,1,1')
    Config.write(cfgfile)
    cfgfile.close()
    
def dhi_load(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath('/home/functionalspinelab/Documents/MATLAB/dhi_release/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/NIfTI_20140122/')
    eng.dhi_load(current_dir + '/' + 'config.ini',nargout=0)
    eng.quit()

def dhi_ia_cal(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nodisplay")
    eng.addpath('/home/functionalspinelab/Documents/MATLAB/dhi_release/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/NIfTI_20140122/')
    #eng.ExecutionEnvironment("cpu")
    eng.dhi_ia_cal(current_dir + '/' + 'config.ini',nargout=0)
    eng.quit()

def dhi_save_ia(patient):
    nii_file = patient
    pathname, filename = os.path.split(nii_file)
    current_dir = pathname
    print(current_dir)
    os.chdir(current_dir)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash -nojvm -nodisplay")
    eng.addpath('/home/functionalspinelab/Documents/MATLAB/dhi_release/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/','/home/functionalspinelab/Documents/MATLAB/dhi_release/Misc/NIfTI_20140122')
    eng.dhi_save_ia(current_dir + '/' + 'config.ini',nargout=0)
    eng.quit()

for control in controls:
    control_ID = 'CSM_C0'+str(control)
    print(control_ID)
    for i in range(len(slices)):

        slice_num = 'slice_'+str(i+1)
        print(slice_num)
        pt_path = os.path.join(control_path,control_ID,'scan_2/dMRI_ZOOMit/',slice_num,'all_volumes/dense/dmri_crop_moco.nii')

        if first_time == 'True':
            #print("Building Config file")
            dhi_config(pt_path)
            #print("Done building config file\n")

            #print("Running DHI Load")
            dhi_load(pt_path)
            #print("Completed DHI Load\n")

        #print("Calculating DHI values")
        dhi_ia_cal(pt_path)
        #print("Completed DHI calculations\n")

        #print("Saving file")
        dhi_save_ia(pt_path)
        #print("Saved files\n")

#sys.exit()
#Run for CSM patients
for csm in csm_patients:
    csm_ID = 'CSM_P0'+str(csm)
    print(csm_ID)
    for i in range(len(slices)):

        slice_num = 'slice_'+str(i+1)
        print(slice_num)
        pt_path = os.path.join(csm_path,csm_ID,'scan_2/dMRI_ZOOMit/',slice_num,'all_volumes/dense/dmri_crop_moco.nii')

        if first_time == 'True':
            dhi_config(pt_path)
            dhi_load(pt_path)

        dhi_ia_cal(pt_path)
        dhi_save_ia(pt_path)

