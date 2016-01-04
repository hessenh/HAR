# Hierachy of Convolutional Neural Networks
# 
# ==============================================================================


class CNN_STATIC_VARS(object):
	''' Variables '''
	CONVERTION_STATIC_DYNAMIC = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2}

	CONVERTION_ORIGINAL = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}

	CONVERTION_STATIC = {6:1, 7:2, 8:3, 16:4, 17:5}
	RE_CONVERTION_STATIC = {1:6, 2:7, 3:8, 4:16, 5:17}

	CONVERTION_DYNAMIC = {1:1, 2:2, 3:3, 4:4, 5:5, 9:6, 10:7, 11:8, 12:9, 13:10, 14:11, 15:12}
	RE_CONVERTION_DYNAIC = {1:1, 2:2, 3:3, 4:4, 5:5, 6:9, 7:10, 8:11, 9:12, 10:13, 11:14, 12:15}
	REMOVE_DYNAMIC_ACTIVITIES = {1:1, 2:2, 3:3, 4:4, 5:5, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15}

	CONVERTION_SHUF_STAND = {1:1,3:2,6:3}
	CONVERTION_SHUF_STAND_INVERSE = {2:2,4:4, 5:5, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
	RE_CONVERTION_SHUF_STAND_WALK = {1:1, 2:3, 3:6}


	''' Methods '''

	''' Get length of convertion list - aka number of labels '''
	def len_convertion_list(self, convertion_list):
		return len(set(convertion_list.values()))

	''' Get config for CNN '''
	def get_config(self, input_size, output_size, iterations, batch_size, model_name):
		return 	{
		   'input_size': input_size, # Number of inputs 
		   'output_size': output_size, # Number of ouptuts
		   'iteration_size': iterations, # Number of training iterations
		   'batch_size': batch_size, # Number of samples in each training iteration (batch)
		   'model_name': model_name
		}

	''' Subject set '''
	def get_subject_set(self):
		TRAIN_SUBJECTS = ["P03","P04","P07","P08","P09","P10","P11","P14","P15","P16","P17","P18","P19","P20","P21"]
		TEST_SUBJECTS = ["P06"]#,"P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
		return [TRAIN_SUBJECTS, TEST_SUBJECTS]
