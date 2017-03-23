#!/usr/bin/python3
#
# models.training_data,py - Class for loading and managing the training data
#
# Frank Blankenburg, Mar. 2017
#

import h5py
import os

    
#--------------------------------------------------------------------------
# CLASS TrainingData
#
# Container keeping the training data sets
#
class TrainingData:
    
    def __init__ (self, file):
        self.sample_size = file.attrs['sample_size']

        self.data = file['data']
        self.labels = file['labels']

        assert len (self.data) == len (self.labels)
        
        self.batch_offset = 0
        
        self.training_set_offset = 0
        self.training_set_size = int (len (self.data) * 0.9)
        
        self.test_set_offset = self.training_set_offset + self.training_set_size
        self.test_set_size = len (self.data) - self.training_set_size
        
        self.segments = file.attrs['segments']
        
            
    def size (self):
        return len (self.data)
    
    def reset (self):
        self.batch_offset = 0
    
    def get_training_data (self):
        return (self.data[self.training_set_offset:self.training_set_offset + self.training_set_size], 
                self.labels[self.training_set_offset:self.training_set_offset + self.training_set_size])
    
    def get_test_data (self):
        return (self.data[self.test_set_offset:self.test_set_offset + self.test_set_size], 
                self.labels[self.test_set_offset:self.test_set_offset + self.test_set_size])
    
    def get_next_batch (self, size):

        data = []
        labels = []
        
        for i in range (self.batch_offset, self.batch_offset + size):
            data.append (self.data[self.training_set_offset + i % self.training_set_size])
            labels.append (self.labels[self.training_set_offset + i % self.training_set_size])
            
        self.batch_offset = (self.batch_offset + size) % self.training_set_size
        
        return (data, labels)
        
    
