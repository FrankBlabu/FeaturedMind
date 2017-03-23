#!/usr/bin/python3
#
# models.cnn_manual.py - Manually setup CNN model
#
# Frank Blankenburg, Mar. 2017
#

import os
import tensorflow as tf

from models.training_data import TrainingData



#--------------------------------------------------------------------------
# Train border detection with a manually constructed multilayer 
# convolutional network
#
def train (args, data):
    
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

    onehot_y_ = tf.one_hot (indices=tf.cast (y_, tf.int32), depth=data.segments + 2)
      
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
    W_fc2 = create_weight_variable ([1024, data.segments + 2])
    b_fc2 = create_bias_variable ([data.segments + 2])
    
    y_conv = tf.add (tf.matmul (h_fc1_drop, W_fc2), b_fc2, name='y_conv')        
    add_variable_summary (y_conv)
    
    result = tf.argmax (y_conv, 1, name='result')

    #
    # Training layers
    #
    with tf.name_scope ('training'):
        cross_entropy = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (labels=onehot_y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)
        correct_prediction = tf.equal (result, tf.cast (y_, tf.int64))
        accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32), name='accuracy')
        
    tf.summary.scalar ('cross_entropy', cross_entropy)
    accuracy_summary = tf.summary.scalar ('accuracy', accuracy)
    
    session = tf.InteractiveSession ()
    
    merged_summary = tf.summary.merge_all ()
    
    if args.log != None:
        train_writer = tf.summary.FileWriter (os.path.join (args.log, 'train'), session.graph)
        test_writer = tf.summary.FileWriter (os.path.join (args.log, 'test'))
     
    session.run (tf.global_variables_initializer ())

    test_data = data.get_test_data ()            
    
    for i in range (args.steps):
        batch = data.get_next_batch (args.batchsize)
        
        if i % 100 == 0:
            train_accuracy = accuracy.eval (feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            test_accuracy = accuracy.eval (feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0})
            print ("    Step {0}, training data accuracy {1:.4f}, test data accuracy {2:.4f}"
                   .format (i, train_accuracy, test_accuracy))

        if args.log != None:    
            summary, _ = session.run ([merged_summary, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            train_writer.add_summary (summary, i)
            
            test_summary = session.run (accuracy_summary, 
                                        feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0})
            test_writer.add_summary (test_summary, i)
        else:
            session.run (train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
            
    test_data = data.get_test_data ()
    test_data_accuracy = accuracy.eval (feed_dict={x: test_data[0], y_: test_data[1], keep_prob: 1.0})
    print ("  Test set accuracy: {0:.4f}".format (test_data_accuracy))
    
    if args.log != None:
        train_writer.close ()
        test_writer.close ()
    
    #
    # Save model if configured
    #
    if args.output != None:
        saver = tf.train.Saver ()
        saver.save (session, os.path.abspath (args.output))

    return session
        
