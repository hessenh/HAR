import pandas as pd
import numpy as np
subject = 'P03'

filepath = '../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/1.5/ORIGINAL/'
files =   [
'Axivity_CHEST_Back_X.csv', 'Axivity_THIGH_Left_Y.csv', 
'Axivity_CHEST_Back_Y.csv', 'Axivity_THIGH_Left_Z.csv', 
'Axivity_CHEST_Back_Z.csv', 'Axivity_THIGH_Left_X.csv']
df_0 = pd.read_csv(filepath+files[0], header=None, sep='\,',engine='python')
df_1 = pd.read_csv(filepath+files[1], header=None, sep='\,',engine='python')
df_2 = pd.read_csv(filepath+files[2], header=None, sep='\,',engine='python')
df_3 = pd.read_csv(filepath+files[3], header=None, sep='\,',engine='python')
df_4 = pd.read_csv(filepath+files[4], header=None, sep='\,',engine='python')
df_5 = pd.read_csv(filepath+files[5], header=None, sep='\,',engine='python')

filepath = '../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/1.5/ORIGINAL/Usability_LAB_All_L.csv'
  

df_labels = pd.read_csv(filepath, header=None, sep='\,',engine='python')
df_labels.columns = ['labels']
df_data = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_labels],axis=1)

CONVERTION_STATIC = {1:1, 2:2, 3:3, 4:4, 5:5, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15}

started = False
for key, value in CONVERTION_STATIC.iteritems():
   df_data =  df_data[df_data['labels'] != value]
df_labels = df_data['labels']
df_data = df_data.drop('labels', 1)


print len(df_data)

print len(df_labels)


con = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
d = []
for act in con:
  if act not in CONVERTION_STATIC:
    d.append(act)
print d

m = []
for i in range(len(df_labels)):
  a = df_labels.iloc[i]
  n =  np.zeros(5)
  print(n)
  print(a)
  n[a-1] = 1
  m.append(n)

s = pd.DataFrame(m)