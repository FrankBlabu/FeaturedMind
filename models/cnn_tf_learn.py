#!/usr/bin/python3
#
# models.cnn_tf_learn.py - CNN setup via tf.learn
#
# Frank Blankenburg, Mar. 2017
#

import os
import tensorflow as tf

from models.training_data import TrainingData
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


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
def train (args, data):
    
    tf.logging.set_verbosity (tf.logging.INFO)
    
    classifier = tf.contrib.learn.Estimator (
        model_fn = build_tf_learn_model, 
        model_dir='/tmp/cnn'
        )
          
    classifier.fit (
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
    
