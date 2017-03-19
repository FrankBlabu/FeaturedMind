#!/usr/bin/python3
#
# train_border_detection.py - Train net to recognize border structures
#                             in feature images
#
# Frank Blankenburg, Mar. 2017
#

import argparse
import h5py

import tensorflow as tf
import numpy as np

from common import Vec2d

from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

    
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
        
    

#--------------------------------------------------------------------------
# Train border detection with a manually constructed multilayer 
# convolutional network
#
def train_manual (config, data):
    
    #
    # Generate variable with given shape and a truncated normal distribution
    #
    def create_weight_variable (shape):
        initial = tf.truncated_normal (shape, stddev=0.1)
        return tf.Variable (initial)
    
    #
    # Generate constant for bias variable with a given shape
    #
    def create_bias_variable (shape):
        initial = tf.constant (0.1, shape=shape)
        return tf.Variable (initial)
    
    #
    # Configure summarizing noded for a variable
    #  
    def add_variable_summary (var):
        with tf.name_scope ('summaries'):
            mean = tf.reduce_mean (var)
            tf.summary.scalar ('mean', mean)
            
            with tf.name_scope ('stddev'):
                stddev = tf.sqrt (tf.reduce_mean (tf.square (var - mean)))                
                tf.summary.scalar ('stddev', stddev)
                
            tf.summary.scalar ('max', tf.reduce_max (var))
            tf.summary.scalar ('min', tf.reduce_min (var))
            tf.summary.histogram ('histogram', var)

    #
    # Input layer, 'None' stand for any size (batch size, in this case)
    #
    with tf.name_scope ('input'):
        x = tf.placeholder (tf.float32, [None, data.sample_size * data.sample_size], name='x')
        y_ = tf.placeholder (tf.float32, [None], name='y_')

    onehot_y_ = tf.one_hot (indices=tf.cast (y_, tf.int32), depth=2)
      
    #
    # Generate 2D image from the continuous input data
    #
    # Size: [Batch size, image width, image height, image depth]
    #
    # If one component is -1 (here: the batch size) it is computed automatically so that
    # the data matches the target shape. 
    #
    x_image = tf.reshape (x, [-1, data.sample_size, data.sample_size, 1], name='image')
    tf.summary.image ('input image', x_image, 10)
    
    #
    # Convolution layer 1 (plus ReLu)
    #
    # Input size : [Batch size, image width, image height, output channels]    
    # Filter size: [Conv. area width, conv. area height, input channels (image channels), output channels (32)]
    # Output size: [Batch size, image width, image height, depth (32)] 
    #
    # So the input image is scanned in 5x5 samples and 32 features are extracted from
    # each sample 
    #    
    W_conv1 = create_weight_variable ([5, 5, 1, 32])
    b_conv1 = create_bias_variable ([32])
    h_conv1 = tf.nn.conv2d (x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    h_relu1 = tf.nn.relu (h_conv1 + b_conv1)
    
    #
    # Max pooling layer 1
    #
    # The window size is [1, 2, 2, 1], so each batch entry and output channel is used, but
    # 2x2 samples of the convoluted image of the last step are max pooled together.
    #
    # Output size: [batch size, image width / 2, image height / 2, depth (32)]
    #
    h_pool1 = tf.nn.max_pool (h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        
    #
    # Convolutional layer 2 (plus ReLu)
    #
    # Input size:  [batch size, image width / 2, image height / 2, input depth (32)]
    # Filter size: [Conv. area width, conv area height, input channels, output depth (64)]
    # Output size: [batch size, image width / 2, image height / 2, output depth (64)]
    #                        
    W_conv2 = create_weight_variable ([5, 5, 32, 64])
    b_conv2 = create_bias_variable ([64])
    h_conv2 = tf.nn.conv2d (h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_relu2 = tf.nn.relu (h_conv2 + b_conv2)
    
    #
    # Max pooling layer 2
    #
    # Output size: [batch size, image width / 4, image height / 4, output depth (64)]
    #
    h_pool2 = tf.nn.max_pool (h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #
    # Full connected layer (plus ReLu)
    #
    # Input size: [batch size, image width / 4, image height / 4, input depth (64)]
    # Ouput size: [batch size, output depth (1024)]
    #            
    W_fc1 = create_weight_variable ([8 * 8 * 64, 1024])
    b_fc1 = create_bias_variable ([1024])
    
    h_pool2_flat = tf.reshape (h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu (tf.matmul (h_pool2_flat, W_fc1) + b_fc1)

    #
    # Dropout of the fully connected layer
    #    
    with tf.name_scope ('dropout'):
        keep_prob = tf.placeholder (tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout (h_fc1, keep_prob)
        
        tf.summary.scalar ('dropout_keep_probability', keep_prob)
    
    #
    # Output layer
    #
    # Reduce to expected classes (has no border/ has a border)
    #
    W_fc2 = create_weight_variable ([1024, 2])
    b_fc2 = create_bias_variable ([2])
    
    y_conv = tf.add (tf.matmul (h_fc1_drop, W_fc2), b_fc2, name='y_conv')        
    add_variable_summary (y_conv)
    

    #
    # Training layers
    #
    with tf.name_scope ('training'):
        cross_entropy = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels=onehot_y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)
        correct_prediction = tf.equal (tf.argmax (y_conv, 1), tf.argmax (onehot_y_, 1))
        accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32), name='accuracy')
        
    tf.summary.scalar ('cross_entropy', cross_entropy)
    tf.summary.scalar ('accuracy', accuracy)
    
    session = tf.InteractiveSession ()
    
    merged_summary = tf.summary.merge_all ()
    
    if args.log != None:
        train_writer = tf.summary.FileWriter (args.log, session.graph)
     
    session.run (tf.global_variables_initializer ())
    
    for i in range (args.steps):
        batch = data.get_next_batch (args.batchsize)
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval (feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print ("    Step {0}, training accuracy {1:.4f}".format (i, train_accuracy))

        if args.log != None:    
            summary, _ = session.run ([merged_summary, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            train_writer.add_summary (summary, i)
        else:
            session.run (train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            
    test_data = data.get_test_data ()
    test_data_accuracy = accuracy.eval (feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0})
    print ("  Test set accuracy: {0:.4f}".format (test_data_accuracy))
    
    if args.log != None:
        train_writer.close ()
    
    #
    # Save model if configured
    #
    if args.output != None:
        saver = tf.train.Saver ()
        saver.save (session, args.output)


#--------------------------------------------------------------------------
# Build model based on tf.learn
#
def build_tf_learn_model (features, labels, mode):
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features, [-1, data.sample_size, data.sample_size, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, sample_size, sample_size, 1]
    # Output Tensor Shape: [batch_size, sample_size, sample_size, 32]
    conv1 = tf.layers.conv2d (
        inputs=input_layer,
        filters=32,           
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
        )

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, sample_size, sample_size, 32]
    # Output Tensor Shape: [batch_size, sample_size / 2, sample_size / 2, 32]
    pool1 = tf.layers.max_pooling2d (inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, sample_size / 2, sample_size / 2, 32]
    # Output Tensor Shape: [batch_size, sample_size / 2, sample_size / 2, 64]
    conv2 = tf.layers.conv2d (
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, sample_size / 2, sample_size / 2, 64]
    # Output Tensor Shape: [batch_size, sample_size / 4, sample_size / 4, 64]
    pool2 = tf.layers.max_pooling2d (inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, sample_size / 4, sample_size / 4, 64]
    # Output Tensor Shape: [batch_size, sample_size / 4 * sample_size / 4 * 64]
    pool2_flat = tf.reshape (pool2, [-1, int ((data.sample_size / 4) * (data.sample_size / 4) * 64)])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, sample_size / 4 * sample_size / 4 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense (inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout (inputs=dense, rate=0.4, 
                                 training=mode is tf.contrib.learn.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense (inputs=dropout, units=2)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != tf.contrib.learn.ModeKeys.INFER:
      onehot_labels = tf.one_hot (indices=tf.cast (labels, tf.int32), depth=2)
      loss = tf.losses.softmax_cross_entropy (onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss (
          loss=loss,
          global_step=tf.contrib.framework.get_global_step (),
          learning_rate=0.001,
          optimizer="SGD")
    
    # Generate Predictions
    predictions = {
        "classes"      : tf.argmax (input=logits, axis=1),
        "probabilities": tf.nn.softmax (logits, name="softmax_tensor")
    }
    
    # Build a ModelFnOps object
    model = model_fn_lib.ModelFnOps (
        mode=mode, 
        predictions=predictions, 
        loss=loss, 
        train_op=train_op
        )
    
    return model


#--------------------------------------------------------------------------
# Train border detection with a multilayer convolutional network
# based on tf.learn
#
def train_tf_learn (config, data):
    
    tf.logging.set_verbosity (tf.logging.INFO)
    
    classifier = tf.contrib.learn.Estimator (
        model_fn = build_tf_learn_model, 
        model_dir='/tmp/cnn'
        )
          
    classifier.fit (
        #input_fn = lambda: data.get_training_data (),
        x=data.get_training_data ()[0],
        y=data.get_training_data ()[1],
        batch_size=config.batchsize,
        steps=config.steps
        )

    metrics = {
      "accuracy": tf.contrib.learn.MetricSpec (metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }
    
    eval_results = classifier.evaluate (
        x=data.get_test_data ()[0],
        y=data.get_test_data ()[1],
        metrics=metrics
        )
    
    print(eval_results)
        
#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',              type=str,               help='Test dataset file name')
parser.add_argument ('-s', '--steps',     type=int, default=1000, help='Number of steps')
parser.add_argument ('-l', '--log',       type=str, default=None, help='Log file directory')
parser.add_argument ('-o', '--output',    type=str, default=None, help='Model output file name')
parser.add_argument ('-b', '--batchsize', type=int, default=50,   help='Number of samples per training batch')

args = parser.parse_args ()

assert args.steps >= 100

#
# Load sample data
#
file = h5py.File (args.file, 'r')
data = TrainingData (file)
    
print ("Training model...")
print ("  Number of samples: ", data.size ())
print ("  Sample size      : ", data.sample_size)

    
#train_manual (args, data)
train_tf_learn (args, data)

file.close ()

