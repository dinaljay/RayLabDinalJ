import pandas as pd
import numpy as np

x = [1,2,3,4,5,6]

#print("x data types is:", type(x))

y = pd.Categorical(x)

print("y data types is:", type(y))

controls = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])
mild_cm_subjects = np.array([1,2,3,4,10,15,16,18,19,21,23,24,26,28,29,31,32,33,36,38,40,42,43,44,45,46])
moderate_cm_subjects = np.array([5,6,9,12,13,14,20,22,25,27,30,34,35,37,39,41,47])

all_cm = np.concatenate((mild_cm_subjects,moderate_cm_subjects),axis=0)

control_ids = ["control"]*len(controls)
csm_ids = ["csm"]*len(all_cm)

all_ids = control_ids+csm_ids

print(all_ids)
print(type(all_ids))