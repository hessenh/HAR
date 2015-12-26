import tensorflow as tf
import input_data_window_large


data = input_data_window_large.read_data_sets("P03","P09")
batch = data.train._data

print batch