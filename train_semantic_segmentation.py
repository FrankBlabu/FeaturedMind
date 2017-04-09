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

from __future__ import print_function

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

smooth = 1.0

def dice_coef (y_true, y_pred):
    y_true_f = K.flatten (y_true)
    y_pred_f = K.flatten (y_pred)
    intersection = K.sum (y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum (y_true_f) + K.sum (y_pred_f) + smooth)


def dice_coef_loss (y_true, y_pred):
    return -dice_coef (y_true, y_pred)


def create_model ():
    inputs = Input ((img_rows, img_cols, 1))
    conv1 = Conv2D (32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D (32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D (pool_size=(2, 2))(conv1)

    conv2 = Conv2D (64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D (64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D (pool_size=(2, 2))(conv2)

    conv3 = Conv2D (128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D (128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D (pool_size=(2, 2))(conv3)

    conv4 = Conv2D (256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D (256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D (pool_size=(2, 2))(conv4)

    conv5 = Conv2D (512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D (512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate ([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D (256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D (256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate ([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D (128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D (128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate ([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D (64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D (64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate ([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D (32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D (32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D (1, (1, 1), activation='sigmoid')(conv9)

    model = Model (inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_and_predict():

    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean
    #imgs_train /= std

    #imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]

    model = create_model ()
    #model_checkpoint = ModelCheckpoint('weights.h5', monitor='loss', save_best_only=True)

    model.fit (imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
               validation_split=0.2)
               #callbacks=[model_checkpoint])

    #model.load_weights('weights.h5')

    #imgs_mask_test = model.predict(imgs_test, verbose=1)


#--------------------------------------------------------------------------
# MAIN
#

#
# Parse command line arguments
#
parser = argparse.ArgumentParser ()

parser.add_argument ('file',                type=str,               help='Test dataset file name')
parser.add_argument ('-e', '--epochs',      type=int, default=50,   help='Number of epochs')
parser.add_argument ('-l', '--log',         type=str, default=None, help='Log file directory')
parser.add_argument ('-o', '--output',      type=str, default=None, help='Model output file name')
parser.add_argument ('-b', '--batchsize',   type=int, default=128,  help='Number of samples per training batch')
parser.add_argument ('-t', '--tensorboard', action='store_true', default=False, help='Open log in tensorboard')

args = parser.parse_args ()

assert not args.tensorboard or args.log

#
# Delete old log files
#
if args.log:
    for root, dirs, files in os.walk (args.log, topdown=False):
        for name in files:
            if name.startswith ('events'):
                os.remove (os.path.join (root, name))

#
# Load sample data
#
file = h5py.File (args.file, 'r')
    
print ("Training model...")
print ("  Number of images: ", data.size ())
    
train_cnn (args, data)

