'''
Create new transitions. 
A transition is an activity that happens between activities. 
Previously we only used one transition and we are now using the activities before and 
after to specify the transition.

Example: A transition between the activity standing and sitting could be called 
transition_stand_sit

We will only modify the label file.

# Old activity list 
1:'walking'	
2:'walking with transition'	
3:'shuffling'	
4:'stairs (ascending)'	
5:'stairs (descending)'	
6:'standing'	
7:'sitting'	
8:'lying'	
9:'transition'	
10:'leaning'	
11:'undefined'	
12:'jumping'	
13:'Dynamic'	
14:'Static'	
15:'Shake'	
16:'picking'	
17:'kneeling'


Transitions numbers: (see transition_info.txt for expl)
[6][7] - transition_stand_sit
[7][6] - transition_sit_stand
[1][1] - transition_walk_walk
[1][7] - transition_walk_sit
[7][1] - transition_sit_walk
[6][6] - transition_stand_stand
[6][8] - transition_stand_lie
[8][6] - transition_lie_stand
[3][7] - transition_shuf_sit
[7][3] - transition_sit_shuf
[7][8] - transition_sit_lie
'''
transitions = {}
transitions["[1][1]"] = 18 # IMPORTANT
transitions["[1][7]"] = 19 # IMPORTANT
#transitions["[3][7]"] = 20
transitions["[6][6]"] = 21 # IMPORTANT
transitions["[6][7]"] = 22 # IMPORTANT
#transitions["[6][8]"] = 23
transitions["[7][1]"] = 24 # IMPORTANT
#transitions["[7][3]"] = 25
transitions["[7][6]"] = 26 # IMPORTANT
#transitions["[7][8]"] = 27
#transitions["[8][6]"] = 28
# new
#transitions["[6][3]"] = 29
transitions["[7][7]"] = 30 # IMPORTANT
#transitions["[6][17]"] = 31
#transitions["[17][6]"] = 32
#transitions["[3][8]"] = 33
#transitions["[8][7]"] = 34
transitions["[8][1]"] = 35 # IMPORTANT
# transitions["[1][8]"] = 36
# transitions["[8][12]"] = 37
# transitions["[3][10]"] = 38
# transitions["[10][6]"] = 39
# transitions["[6][10]"] = 40
# transitions["[7][17]"] = 41
# transitions["[17][7]"] = 42
# transitions["[6][1]"] = 43
# transitions["[1][3]"] = 44
# transitions["[7][16]"] = 45
# transitions["[16][7]"] = 46
# transitions["[16][6]"] = 47
# transitions["[8][3]"] = 48
# transitions["[3][3]"] = 49
# transitions["[1][16]"] = 50
# transitions["[16][3]"] = 51
# transitions["[17][3]"] = 52
# transitions["[3][17]"] = 53
# transitions["[8][8]"] = 54
# transitions["[3][1]"] = 55
# transitions["[10][10]"] = 56
# transitions["[6][16]"] = 57
# transitions["[3][6]"] = 58
# transitions["[1][6]"] = 59
# transitions["[7][10]"] = 60
# transitions["[10][7]"] = 61
# transitions["[16][1]"] = 62






import pandas as pd
import numpy as np


# Import label
subjects = ["P03","P04","P06","P07","P08","P09","P10","P11","P14","P15","P16","P17","P18","P19","P20","P21"]

for subject in subjects:
	print 'Changing subject ', subject
	filepath = '../../Prosjektoppgave/Notebook/data/'+subject+'/RAW_SIGNALS/'+subject+'_Usability_LAB_All.csv'

	df = pd.read_csv(filepath, header=None, sep='\,',engine='python')

	old_list =  df.as_matrix(columns=None)

	new_list = old_list.copy()

	# Used when counting occurences of transition
	transition_counter = []

	start = 0

	def change_list(i,j):
		# Get the values before and after transition
		before = old_list[i-1]
		after = old_list[j]

		# Used in dictionary
		before_after = str(old_list[i-1])+str(old_list[j])


		transition_counter.append(before_after)
		# Change all transition values according to before and after value
		for k in range(i,j):
			# Check to see if transition is included
			if before_after in transitions:
				# Change transition
				new_list[k] = [transitions[before_after]]
			

	for i in range(len(old_list)):
		# Find start of transition
		if old_list[i-1] != [9] and old_list[i] == [9]:
			#Start of transition
			start = i
			# Find end of transition
			for j in range(i,len(old_list)):
				if old_list[j] != [9]:
					# Count occurance of transition
					before_after =  str(old_list[i-1])+str(old_list[j])
					# Change values in new list
					change_list(i,j)
					break


	#print df[215:230]

	df = pd.DataFrame(new_list)

	#print df[215:230]


	''' Count different occurances of transition '''
	c =  dict((i, transition_counter.count(i)) for i in transition_counter)
	for key, value in c.iteritems() :
	    print key, value


	''' Save new DataFrame '''
	filepath = '../../Prosjektoppgave/Notebook/data/'+subject+'/RAW_SIGNALS/'+subject+'_Usability_LAB_All_NEW_TRANSITION.csv'

	df.to_csv(filepath,  header=None, index=False)










