import input_data_window_large
import CNN

# Subjects
train_subjects = ["P03","P04"]#,"P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
test_subjects = ["P11"]#,"P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
subject_set = [train_subjects, test_subjects]



''' STATIC VS DYNAMIC NETWORK'''

# Convert labels from original label to new label 
'''Static vs Dynamic'''
convertion = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2}
number_of_labels = len(set(convertion.values()))
data_set = input_data_window_large.read_data_sets(subject_set, number_of_labels, convertion, None)
# Config
config = {
   'input_size': 900, # Number of inputs 
   'output_size': number_of_labels, # Number of ouptuts
   'iteration_size': 200, # Number of training iterations
   'batch_size': 100 # Number of samples in each training iteration (batch)
}

cnn = CNN.CNN_TWO_LAYERS(data_set, config)
#cnn.train_network()
#cnn.save_model('ds')
cnn.load_model('ds')
cnn.test_network(100)

'''Original'''
convertion_2 = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
number_label_2 = len(set(convertion_2.values()))
data_set_2 = input_data_window_large.read_data_sets(subject_set, number_label_2, convertion_2, None)
# Config
config = {
   'input_size': 900, # Number of inputs 
   'output_size': number_label_2, # Number of ouptuts
   'iteration_size': 200, # Number of training iterations
   'batch_size': 100 # Number of samples in each training iteration (batch)
}

cnn2 = CNN.CNN_TWO_LAYERS(data_set_2, config)
#cnn2.train_network()
#cnn2.save_model('original')
cnn2.load_model('original')
cnn2.test_network(100)

