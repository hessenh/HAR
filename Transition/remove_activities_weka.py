import pandas as pd
from os import path
import os
from os import listdir

filepath = '../../Prosjektoppgave/Notebook/data/1.5.arff'

#REMOVE"
activities = [""]

with open(filepath) as f:
    result = f.readlines()


new = []

for i in range(0,100):
	t = result[i].split()
	print t
#with open(RESULT_PATH, 'w') as file:
#	file.writelines(result)
