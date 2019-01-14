import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D, Cropping2D
from keras.layers.core import Dropout
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
    elif model_type == 'base':
        model = base_model(image_dim, nlabels, nK, n_dil, kernel_size, drop_out, activation_hidden, activation_output, verbose=1)

    return model

def compile_and_run(target_dir, model, model_name, model_type, history_fn, X_train,  Y_train, X_validate, Y_validate,  nb_epoch, batch_size, nlabels, loss, verbose=1, metric="accuracy", lr=0.005, nGPU=1):

    #set compiler
    ada = keras.optimizers.Adam(0.0001)

    #set checkpoint filename
    checkpoint_fn = str(os.path.join(target_dir, 'model', str(os.path.basename(model_name).split('.')[0]) +"_checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"))

    #create checkpoint callback for model
    checkpoint = ModelCheckpoint(checkpoint_fn, monitor='val_loss', verbose=0, save_best_only=True, mode='max')

    #compile the model
    #model.compile(loss = , optimizer=ada, metrics=[metric])
    print("Compiling model {}...".format(model_name))

    if 'autoencoder' in model_type:

        model.compile(loss=loss, optimizer=Adam())

    elif 'base' in model_type:

        model.compile(loss=loss, optimizer=Adam())


    print("Training size: {}\n Validation size: {}\n".format(X_train.shape, X_validate.shape))

    # train model
    # augmentation generator
    aug = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.01,
                             height_shift_range=0.01,
                             zoom_range=0.3,
                             fill_mode="nearest")
    #if nGPU > 1:
    if 'autoencoder' in model_type:
        history = model.fit_generator(
                  aug.flow(X_train,
                  X_train,
                  batch_size = batch_size*nGPU),
                  validation_data= (X_validate, X_validate),
                  steps_per_epoch= len(X_train) // (batch_size * nGPU),
                  epochs= nb_epoch,
                  callbacks= [checkpoint])
                  #callbacks= [TensorBoard(log_dir='/home/spinney/scripts/python/MRI_Deep_Learning/logs/autoencoder')])

    elif 'base' in model_type:
        history = model.fit_generator(
            aug.flow(X_train,
                     Y_train,
                     batch_size=batch_size * nGPU),
                    validation_data=(X_validate, X_validate),
                    steps_per_epoch=len(X_train) // (batch_size * nGPU),
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