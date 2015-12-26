import input_data_window_large

train_subjects = ["P03"]#,"P04","P06","P07","P08","P09","P10","P14","P15","P16","P17","P18","P19","P20","P21"]
test_subjects = ["P11"]
subject_set = [train_subjects, test_subjects]

conv = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:2, 17:2}
number_of_different_labels = len(set(conv.values()))

data_set = input_data_window_large.read_data_sets(subject_set, number_of_different_labels, conv)

print data_set.train._labels

