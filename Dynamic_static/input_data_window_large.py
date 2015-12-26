"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import pandas as pd
import numpy as np

def extract_merged_axes(subject):
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

  df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5],axis=1)
  return df.as_matrix(columns=None)

def extract_data(subjects):
  print('Extracting data from', subjects)

  train_data = extract_merged_axes(subjects[0])

  for i in range(1,len(subjects)):
    subject_data = extract_merged_axes(subjects[i])

    train_data = np.concatenate((train_data,subject_data ), axis=0)

  return train_data

def extract_merged_labels(subject, output_size, change_labels):
  filepath = '../../Prosjektoppgave/Notebook/data/'+subject+'/DATA_WINDOW/1.5/ORIGINAL/Usability_LAB_All_L.csv'
  

  df = pd.read_csv(filepath, header=None, sep='\,',engine='python')

  # Convert from nunber to readable format: 
  # From: [2]
  # To: [0,1,0,0,0,0,0,0,0,0,0]
  m = []
  for i in range(len(df)):
    a = df.iloc[i]
    if change_labels:
      a = change_labels[a.values[0]]
    n =  np.zeros(output_size)
    n[a-1] = 1
    m.append(n)

  s = pd.DataFrame(m)


  return s.values


def extract_labels(subjects, output_size, change_labels):
  print('Extracting label from', subjects)
  train_label = extract_merged_labels(subjects[0], output_size, change_labels)

  for i in range(1,len(subjects)):
    subject_label = extract_merged_labels(subjects[i], output_size, change_labels)

    train_label = np.concatenate((train_label,subject_label ), axis=0)

  return train_label

class DataSet(object):

  def __init__(self, data, labels):
    self._num_examples = data.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  def label_size(self):
      return len(self._labels)
  

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
   
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._data = self._data[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._labels[start:end]

def read_data_sets(subjects_set, output_size, change_labels, load_model):
  training_subjects = subjects_set[0]
  test_subjects = subjects_set[1]
  
  class DataSets(object):
    pass
  data_sets = DataSets()

  # If the model is for testing
  if load_model:
    # Testing data and labels
    test_data = extract_data(test_subjects)
    test_labels = extract_labels(test_subjects, output_size, change_labels)

    # Define testing data sets
    data_sets.test = DataSet(test_data, test_labels)

  else:
     # Training data and labels
    train_data = extract_data(training_subjects)
    train_labels = extract_labels(training_subjects, output_size, change_labels)

    # Testing data and labels
    test_data = extract_data(test_subjects)
    test_labels = extract_labels(test_subjects, output_size, change_labels)

    # Define training and testing data sets
    data_sets.train = DataSet(train_data, train_labels)
    data_sets.test = DataSet(test_data, test_labels)

  return data_sets
