#!/usr/bin/python3
#
# train_semantic_segmentation.py - Train deep learning network for finding segments
#
# Frank Blankenburg, Apr. 2017
# Based on: https://github.com/nicolov/segmentation_keras
#

from keras import optimizers
from keras.layers import Activation, Reshape, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential

def get_frontend (input_width, input_height) -> Sequential:
    model = Sequential ()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add (Convolution2D (64, (3, 3), activation='relu', name='conv1_1', input_shape= (input_width, input_height, 3)))
    model.add (Convolution2D (64, (3, 3), activation='relu', name='conv1_2'))
    model.add (MaxPooling2D ((2, 2), strides=(2, 2)))

    model.add (Convolution2D (128, (3, 3), activation='relu', name='conv2_1'))
    model.add (Convolution2D (128, (3, 3), activation='relu', name='conv2_2'))
    model.add (MaxPooling2D ((2, 2), strides=(2, 2)))

    model.add (Convolution2D (256, (3, 3), activation='relu', name='conv3_1'))
    model.add (Convolution2D (256, (3, 3), activation='relu', name='conv3_2'))
    model.add (Convolution2D (256, (3, 3), activation='relu', name='conv3_3'))
    model.add (MaxPooling2D ((2, 2), strides=(2, 2)))

    model.add (Convolution2D (512, (3, 3), activation='relu', name='conv4_1'))
    model.add (Convolution2D (512, (3, 3), activation='relu', name='conv4_2'))
    model.add (Convolution2D (512, (3, 3), activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add (Convolution2D (512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add (Convolution2D (512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add (Convolution2D (512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add (Convolution2D (4096, (7, 7), dilation_rate=(4, 4), activation='relu', name='fc6'))
    model.add (Dropout (0.5))
    model.add (Convolution2D (4096, (1, 1), activation='relu', name='fc7'))
    model.add (Dropout (0.5))
    # Note: this layer has linear activations, not ReLU
    model.add (Convolution2D (21, (1, 1), activation='linear', name='fc-final'))

    # model.layers[-1].output_shape == (None, 16, 16, 21)
    return model


def add_softmax (model: Sequential) -> Sequential:
    """ Append the softmax layers to the frontend or frontend + context net. """
    # The softmax layer doesn't work on the (width, height, channel)
    # shape, so we reshape to (width*height, channel) first.
    # https://github.com/fchollet/keras/issues/1169
    _, curr_width, curr_height, curr_channels = model.layers[-1].output_shape

    model.add (Reshape ((curr_width * curr_height, curr_channels)))
    model.add (Activation ('softmax'))
    # Technically, we need another Reshape here to reshape to 2d, but TF
    # then complains when batch_size > 1. We're just going to reshape in numpy.
    # model.add(Reshape((curr_width, curr_height, curr_channels)))

    return model


def add_context (model: Sequential) -> Sequential:
    """ Append the context layers to the frontend. """
    model.add (ZeroPadding2D (padding=(33, 33)))
    model.add (Convolution2D (42, (3, 3), activation='relu', name='ct_conv1_1'))
    model.add (Convolution2D (42, (3, 3), activation='relu', name='ct_conv1_2'))
    model.add (Convolution2D (84, (3, 3), dilation_rate=(2, 2), activation='relu', name='ct_conv2_1'))
    model.add (Convolution2D (168, (3, 3), dilation_rate=(4, 4), activation='relu', name='ct_conv3_1'))
    model.add (Convolution2D (336, (3, 3), dilation_rate=(8, 8), activation='relu', name='ct_conv4_1'))
    model.add (Convolution2D (672, (3, 3), dilation_rate=(16, 16), activation='relu', name='ct_conv5_1'))
    model.add (Convolution2D (672, (3, 3), activation='relu', name='ct_fc1'))
    model.add (Convolution2D (21, (1, 1), name='ct_final'))

    return model


def train ():

    # Create image generators for the training and validation sets. Validation has
    # no data augmentation.
    #transformer_train = RandomTransformer (horizontal_flip=True, vertical_flip=True)
    #datagen_train = SegmentationDataGenerator (transformer_train)

    #transformer_val = RandomTransformer (horizontal_flip=False, vertical_flip=False)
    #datagen_val = SegmentationDataGenerator (transformer_val)

    #train_desc = '{}-lr{:.0e}-bs{:03d}'.format(
    #    time.strftime("%Y-%m-%d %H:%M"),
    #    learning_rate,
    #    batch_size)
    #checkpoints_folder = 'trained/' + train_desc
    
    #model_checkpoint = callbacks.ModelCheckpoint(
    #    checkpoints_folder + '/ep{epoch:02d}-vl{val_loss:.4f}.hdf5',
    #    monitor='loss')
    #tensorboard_cback = callbacks.TensorBoard(
    #    log_dir='{}/tboard'.format(checkpoints_folder),
    #    histogram_freq=0,
    #    write_graph=False,
    #    write_images=False)
    #csv_log_cback = callbacks.CSVLogger(
    #    '{}/history.log'.format(checkpoints_folder))
    #reduce_lr_cback = callbacks.ReduceLROnPlateau(
    #    monitor='val_loss',
    #    factor=0.2,
    #    patience=5,
    #    verbose=1,
    #    min_lr=0.05 * learning_rate)

    model = add_softmax (get_frontend (500, 500))

    model.compile (loss='sparse_categorical_crossentropy',
                   optimizer=optimizers.SGD (),#(lr=0.5, momentum=0.9),
                   metrics=['accuracy'])

    #model.fit_generator(
    #    datagen_train.flow_from_list(
    #        train_img_fnames,
    #        train_mask_fnames,
    #        shuffle=True,
    #        batch_size=batch_size,
    #        img_target_size=(500, 500),
    #        mask_target_size=(16, 16)),
    #    samples_per_epoch=len(train_basenames),
    #    nb_epoch=20,
    #    validation_data=datagen_val.flow_from_list(
    #        val_img_fnames,
    #        val_mask_fnames,
    #        batch_size=8,
    #        img_target_size=(500, 500),
    #        mask_target_size=(16, 16)),
    #    nb_val_samples=len(val_basenames),
    #    callbacks=[
    #        model_checkpoint,
    #        tensorboard_cback,
    #        csv_log_cback,
    #        reduce_lr_cback,
    #        skipped_report_cback,
    #    ])

if __name__ == '__main__':
    train ()
