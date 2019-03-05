import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D, Cropping2D
from keras.layers.core import Dropout, Flatten
from keras.layers import LeakyReLU, MaxPooling2D, concatenate, Conv2DTranspose, Concatenate
from keras.activations import relu
from keras.callbacks import History, ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from sklearn.utils import shuffle
import numpy as np
#from predict import save_image
# from custom_loss import *
# from models.neurotech_models import *
from math import sqrt
from helpers import *
import json


def cnn_binary_classifier_2(image_dim,verbose=1):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_dim[0], image_dim[1], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    if verbose > 0:
        print(model.summary())

    return model



def cnn_binary_classifier(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))

    #pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(8, (5, 5), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (5, 5), activation='relu', padding='same')(conv1) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv3 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (5, 5), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    drop1 = Dropout(0.2)(conv3)
    flat = Flatten()(drop1)
    dense1 = Dense(128, activation='relu')(flat)
    drop2 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(drop2)
    # model.add(Dropout(0.3))
    dense3 = Dense(2, activation='softmax')(dense2)

    classifier = Model(input_img, dense3)

    if verbose > 0:
        print(classifier.summary())

    return classifier


def cnn_autoencoder_6(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))
    k = 3
    #pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(32, (k, k), activation='relu', padding='same')(input_img)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (k, k), activation='relu', padding='same')(conv1) #28 x 28 x 32
    #conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, (k, k), activation='relu', padding='same')(conv1) #14 x 14 x 64
    #conv2 = BatchNormalization()(conv2)
    conv3 = Conv2D(128, (k, k), activation='relu', padding='same')(conv2) #7 x 7 x 128 (small and thick)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (k, k), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(128, (k, k), activation='relu', padding='same')(conv3) #7 x 7 x 128
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (k, k), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv5 = Conv2D(32, (k, k), activation='relu', padding='same')(conv4) # 14 x 14 x 64
    #conv5 = BatchNormalization()(convk)
    conv5 = Conv2D(16, (k, k), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)

    decoded = Conv2D(1, (k, k), activation='linear', padding='same')(conv5) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder



def cnn_autoencoder_5(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))
    k = 3
    #pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(8, (k, k), activation='relu', padding='same')(input_img)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (k, k), activation='relu', padding='same')(conv1) #28 x 28 x 32
    #conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32, (k, k), activation='relu', padding='same')(conv1) #14 x 14 x 64
    #conv2 = BatchNormalization()(conv2)
    conv3 = Conv2D(128, (k, k), activation='relu', padding='same')(conv2) #7 x 7 x 128 (small and thick)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (k, k), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (k, k), activation='relu', padding='same')(conv3) #7 x 7 x 128
    #conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(32, (k, k), activation='relu', padding='same')(conv4)
    #conv4 = BatchNormalization()(conv4)
    conv5 = Conv2D(16, (k, k), activation='relu', padding='same')(conv4) # 14 x 14 x 64
    #conv5 = BatchNormalization()(convk)
    conv5 = Conv2D(8, (k, k), activation='relu', padding='same')(conv5)
    #conv5 = BatchNormalization()(conv5)

    decoded = Conv2D(1, (k, k), activation='linear', padding='same')(conv5) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder


def cnn_autoencoder_4(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))

    pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pad1) #28 x 28 x 32
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    #conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    #conv4 = BatchNormalization()(conv4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #conv5 = BatchNormalization()(conv5)
    up1 = UpSampling2D((2,2))(conv5) # 14 x 14 x 128
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    #conv6 = BatchNormalization()(conv6)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    #conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    #conv8 = BatchNormalization()(conv8)
    up2 = UpSampling2D((2,2))(conv8) # 28 x 28 x 64
    crop1 = Cropping2D(cropping=(padSize,padSize))(up2)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(crop1)
    conv9 = BatchNormalization()(conv9)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv9) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder



def cnn_autoencoder_2(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))

    pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pad1) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    crop1 = Cropping2D(cropping=(padSize,padSize))(up2)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(crop1) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder

def cnn_autoencoder_3(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))

    pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pad1) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2) #14 x 14 x 32
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)


    #decoder
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6) #7 x 7 x 128
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    up1 = UpSampling2D((2,2))(conv8) # 14 x 14 x 128
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)
    conv11 = BatchNormalization()(conv11)
    up2 = UpSampling2D((2,2))(conv11) # 28 x 28 x 64
    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv12 = BatchNormalization()(conv12)
    crop1 = Cropping2D(cropping=(padSize,padSize))(conv12)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(crop1) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder


def cnn_autoencoder(image_dim,verbose=1):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], 1))

    pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pad1) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)


    #decoder
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    crop1 = Cropping2D(cropping=(padSize,padSize))(up2)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(crop1) # 28 x 28 x 1

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder

#
# # TODO: This CNN requires input layer to be div. by 4 because 2 maxpool layers
# def cnn_autoencoder(image_dim,verbose=1):
#     #encoder
#     #input = 28 x 28 x 1 (wide and thin)
#
#     padSize = int(image_dim[0] % 4 / 2)
#
#     input_img = Input(shape=(image_dim[0], image_dim[1], 1))
#
#     pad1 = ZeroPadding2D(padding=(padSize,padSize))(input_img)
#
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(pad1) #28 x 28 x 32
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#
#
#     #decoder
#     conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)
#     up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
#     crop1 = Cropping2D(cropping=(padSize,padSize))(up2)
#
#     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(crop1) # 28 x 28 x 1
#
#     autoencoder = Model(input_img, decoded)
#
#     if verbose > 0:
#         print(autoencoder.summary())
#
#     return autoencoder

def sparse_autoencoder(inShape, kSize, activation, loss, l1Reg=0.05, verbose=1):

    # this is the size of our encoded representations
    #encoding_dim = kSize  # 32 floats -> compression of factor 8, assuming the input is 125 floats

    # this is our input placeholder
    input_img = Input(shape=(inShape[0],inShape[1],1))

    # "encoded" is the encoded representation of the input
    encoded = Dense(kSize, activation=activation,
                    activity_regularizer=regularizers.l1(0.05))(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(inShape, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    ## TOOLS: this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (kSize-dimensional) input
    encoded_input = Input(shape=(kSize,))

    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    ##

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder


def base_model(image_dim, nlabels, nK, n_dil, kernel_size, drop_out, activation_hidden, activation_output, verbose=1):
    print("N Labels:", nlabels)
    print("Drop out:", drop_out)
    print("Number of Dilations:", n_dil)
    print("Activation hidden:", activation_hidden)
    print("Activation output:", activation_output)
    nK = [int(i) for i in nK.split(",")]
    if n_dil == None:
        n_dil = [1] * len(nK)
    else:
        n_dil = [int(i) for i in n_dil.split(",")]

    IN = CONV = Input(shape=(image_dim[0], image_dim[1], 1))
    n_layers = int(len(nK))
    kDim = [kernel_size] * n_layers

    for i in range(n_layers):
        print("Layer:", i, nK[i], kDim[i], n_dil[i])
        CONV = Conv2D(nK[i], kernel_size=[kDim[i], kDim[i]], dilation_rate=(n_dil[i], n_dil[i]),
                      activation=activation_hidden, padding='same')(CONV)
        CONV = Dropout(drop_out)(CONV)

    OUT = Conv2D(nlabels, kernel_size=[1, 1], activation=activation_output, padding='same')(CONV)
    model = keras.models.Model(inputs=[IN], outputs=OUT)
    if verbose > 0:
        print(model.summary())

    return model


def build_model(image_dim, nlabels,nK, n_dil, kernel_size, drop_out, model_type, activation_hidden, activation_output, loss, verbose=0):
    if model_type == 'cnn-autoencoder':
       model = cnn_autoencoder(image_dim)
    elif model_type == 'cnn-autoencoder-2':
       model = cnn_autoencoder_2(image_dim)
    elif model_type == 'cnn-autoencoder-3':
       model = cnn_autoencoder_3(image_dim)
    elif model_type == 'cnn-autoencoder-4':
       model = cnn_autoencoder_4(image_dim)
    elif model_type == 'cnn-autoencoder-5':
       model = cnn_autoencoder_5(image_dim)
    elif model_type == 'cnn-autoencoder-6':
       model = cnn_autoencoder_6(image_dim)
    elif model_type == 'cnn-binary-classifier':
       model = cnn_binary_classifier(image_dim)
    elif model_type == 'cnn-binary-classifier-2':
       model = cnn_binary_classifier_2(image_dim)
    elif model_type == 'base':
        model = base_model(image_dim, nlabels, nK, n_dil, kernel_size, drop_out, activation_hidden, activation_output, verbose=1)

    return model

def compile_and_run(target_dir, model, model_name, model_type, history_fn, X_train,  Y_train, X_validate, Y_validate,  nb_epoch, batch_size, nlabels, loss, verbose=0, metric="accuracy", lr=0.005, nGPU=1):

    #set compiler
    ada = keras.optimizers.Adam(0.01)

    #set checkpoint filename
    checkpoint_fn = str(os.path.join(target_dir, 'model', str(os.path.basename(model_name).split('.')[0]) +"_checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"))

    #create checkpoint callback for model
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=0, save_best_only=True, mode='max')

    if nGPU == 0:
        steps_per_epoch = batch_size
        nGPU = 1
    elif nGPU == 1:
        steps_per_epoch = len(X_train) // (batch_size*16)
    else:
        steps_per_epoch = len(X_train) // (batch_size * nGPU)

    #compile the model
    #model.compile(loss = , optimizer=ada, metrics=[metric])
    print("Compiling model {}...".format(model_name))

    if 'autoencoder' in model_type:

        model.compile(loss=loss, optimizer=Adam(0.01))

    elif 'classifier' in model_type:

        model.compile(loss=loss, optimizer=Adam(), metrics=[metric])


    print("Training size: {}\n Validation size: {}\n".format(X_train.shape, X_validate.shape))

    # train model
    # augmentation generator
    aug = ImageDataGenerator(rotation_range=1,
                             #width_shift_range=0.01,
                             #height_shift_range=0.01,
                             zoom_range=0.001,
                             fill_mode="nearest")
    #if nGPU > 1:
    if 'autoencoder' in model_type:
        history = model.fit_generator(
                  aug.flow(X_train,
                  X_train,
                  batch_size = batch_size*nGPU),
                  validation_data= (X_validate, X_validate),
                  steps_per_epoch= steps_per_epoch,
                  epochs= nb_epoch,
                  callbacks= [checkpoint])
                  #callbacks= [TensorBoard(log_dir='/home/spinney/scripts/python/MRI_Deep_Learning/logs/autoencoder')])

    elif 'classifier' in model_type:
        history = model.fit_generator(
            aug.flow(X_train,
                     Y_train,
                     batch_size=batch_size * nGPU),
                    validation_data=(X_validate, Y_validate),
                    steps_per_epoch=steps_per_epoch,
                    epochs=nb_epoch,
                    callbacks=[checkpoint])
    #
    # else:
    #
    #     history = model.fit_generator(
    #         aug.flow(X_train,
    #                  X_train,
    #                  batch_size=batch_size * nGPU),
    #                  validation_data=(X_validate, X_validate),
    #                  steps_per_epoch=len(X_train) // (batch_size * nGPU),
    #                  epochs=nb_epoch,
    #                  callbacks=[checkpoint])
        # history = model.fit(X_train,
        #                     X_train,
        #                     validation_data=(X_validate, X_validate),
        #                     epochs=nb_epoch,
        #                     batch_size=batch_size,
        #                     callbacks=[checkpoint])

    # save model
    model.save(model_name)

    with open(history_fn, 'w+') as fp: json.dump(history.history, fp)

    return [model, history]


# ############################### visualize recreation
# # encode and decode some digits
# # note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
#
# # use Matplotlib (don't ask)
# import matplotlib.pyplot as plt
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()