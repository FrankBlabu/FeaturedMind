#!/usr/bin/python3
#
# models.training_data,py - Class for loading and managing the training data
#
# Frank Blankenburg, Mar. 2017
#

from enum import Enum

    
#--------------------------------------------------------------------------
# CLASS TrainingData
#
# Container keeping the training data sets
#
class TrainingData:
    
    class Field (Enum):
        DATA    = 0
        LABELS  = 1
        CLASSES = 2
    
    def __init__ (self, file):
        
        self.sample_size = file.attrs['sample_size']

        data    = file['data']
        labels  = file['labels']
        classes = file['classes']

        assert len (data) == len (labels)
        assert len (data) == len (classes)
        
        training_set_offset = 0
        training_set_size = int (len (data) * 0.9)
                
        self.training_data = {
            TrainingData.Field.DATA    : data[training_set_offset:training_set_offset + training_set_size],
            TrainingData.Field.LABELS  : labels[training_set_offset:training_set_offset + training_set_size],
            TrainingData.Field.CLASSES : classes[training_set_offset:training_set_offset + training_set_size]
            }

        test_set_offset = training_set_offset + training_set_size
        test_set_size = len (data) - training_set_size
        
        self.test_data = {
            TrainingData.Field.DATA    : data[test_set_offset:training_set_offset + test_set_size],
            TrainingData.Field.LABELS  : labels[test_set_offset:training_set_offset + test_set_size],
            TrainingData.Field.CLASSES : classes[test_set_offset:training_set_offset + test_set_size]
            }
        
        assert len (self.training_data) == len (TrainingData.Field)
        assert len (self.test_data) == len (TrainingData.Field)
                
    def size (self):
        return len (self.training_data[TrainingData.Field.DATA]) + len (self.test_data[TrainingData.Field.DATA])
    
    def get_training_data (self, field):
        assert field in self.training_data
        return self.training_data[field]
    
    def get_test_data (self, field):
        assert field in self.test_data
        return self.test_data[field]
    
    