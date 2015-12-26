import pandas as pd
from os import path
import os
from os import listdir

SUBJECTS = ["P03","P04","P06","P07","P08","P09","P10","P11"]
#SUBJECTS = ["P03","P04","P06","P07","P08","P09","P10","P11","P14","P15","P16","P17","P18","P19","P20","P21"]

START_SUB = SUBJECTS[0]
WINDOW_SIZE = "1.0"
RESULT_PATH = 'FEATURES.csv'



# FEATURE
Start_feature = '../Prosjektoppgave/Notebook/data/'+START_SUB+'/FEATURES/'+WINDOW_SIZE+'/FEATURES.csv'

l = 0
print Start_feature
with open(Start_feature) as f:
    result = f.readlines()

l = len(result)
with open(RESULT_PATH, 'w') as file:
	file.writelines(result)

#DATA_INDEX = result.index("@data\n")

for i in range(1,len(SUBJECTS)):
	PATH = '../Prosjektoppgave/Notebook/data/'+SUBJECTS[i]+'/FEATURES/'+WINDOW_SIZE+'/FEATURES.csv'
	with open(PATH) as f:
		result = f.readlines()


	with open(RESULT_PATH, 'a') as file:
		file.writelines(result[1:])


### LABEL
Start_label = '../Prosjektoppgave/Notebook/data/'+START_SUB+'/DATA_WINDOW/'+WINDOW_SIZE+'/ORIGINAL/Usability_LAB_All_L.csv'
RESULT_LABEL = 'LABEL.csv'
with open(Start_label) as f:
    result = f.readlines()

with open(RESULT_LABEL, 'w') as file:
	file.writelines(result)

for i in range(1,len(SUBJECTS)):
	PATH = '../Prosjektoppgave/Notebook/data/'+SUBJECTS[i]+'/DATA_WINDOW/'+WINDOW_SIZE+'/ORIGINAL/Usability_LAB_All_L.csv'
	with open(PATH) as f:
		result = f.readlines()


	with open(RESULT_LABEL, 'a') as file:
		file.writelines(result)




import pandas as pd  
import numpy as np

df = pd.read_csv('LABEL.csv', header=None, sep='\ ')


a = df.iloc[0]
print a
n =  np.zeros(16)
n[a-1] = 1

m = []
for i in range(len(df)):
	a = df.iloc[i]
	n =  np.zeros(17)
	n[a-1] = 1
	m.append(n)

s = pd.DataFrame(m)

s.to_csv('LABEL.csv',  header=None, index=False)
