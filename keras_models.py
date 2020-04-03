import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, Conv3D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D, Cropping2D
from keras.layers.core import Dropout, Flatten
from keras.layers import LeakyReLU, MaxPooling2D, MaxPooling3D,concatenate, Conv2DTranspose, Concatenate
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


def encoder(input_img,k):

    conv1 = Conv2D(32, (k, k), activation='relu', padding='same')(input_img)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (k, k), activation='relu', padding='same')(conv1)  # 28 x 28 x 32
    # conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(128, (k, k), activation='relu', padding='same')(conv1)  # 14 x 14 x 64
    # conv2 = BatchNormalization()(conv2)
    conv3 = Conv2D(256, (k, k), activation='relu', padding='same')(conv2)  # 7 x 7 x 128 (small and thick)
    # conv3 = BatchNormalization()(conv3)
   # conv3 = Conv2D(256, (k, k), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)

    return conv3

def decoder(conv3,k):
    # decoder
    conv4 = Conv2D(128, (k, k), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (k, k), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    conv5 = Conv2D(32, (k, k), activation='relu', padding='same')(conv4)  # 14 x 14 x 64
    # conv5 = BatchNormalization()(convk)
    conv5 = Conv2D(16, (k, k), activation='relu', padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    decoded = Conv2D(1, (k, k), activation='linear', padding='same')(conv5)  # 28 x 28 x 1

    return decoded

def fc(encoded):
    drop1 = Dropout(0.2)(encoded)
    flat = Flatten()(drop1)
    dense1 = Dense(128, activation='relu')(flat)
    drop2 = Dropout(0.2)(dense1)
    dense2 = Dense(64, activation='relu')(drop2)
    drop3 = Dropout(0.2)(dense2)
    out = Dense(2, activation='softmax')(drop3)

    return out

def cnn_binary_classifier(image_dim,verbose=1):

    if True:
        autoencoder = load_model('/home/spinney/scripts/python/MRI_Deep_Learning/processed/model/ac_NVY2.hdf5')

    input_img = Input(shape=(image_dim[0], image_dim[1], image_dim[2], 1))
    k = 3
    encoded = encoder(input_img,k)
    out = fc(encoded)

    classifier = Model(input_img, out)

    for l1, l2 in zip(classifier.layers[:5], autoencoder.layers[:5]):
        l1.set_weights(l2.get_weights())

    for layer in classifier.layers[:5]:
        layer.trainable = False


    if verbose > 0:
        print(classifier.summary())

    return classifier


def cnn_3D_classifier(image_dim,num_classes,verbose=1):

    model = Sequential()
    model.add(
        Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_dim[0],image_dim[1],image_dim[2],1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2)))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2,2,2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def cnn_autoencoder(image_dim,verbose=1):

    padSize = int(image_dim[0] % 4 / 2)

    input_img = Input(shape=(image_dim[0], image_dim[1], image_dim[2],1))
    k = 3

    encoded = encoder(input_img,k)
    decoded = decoder(encoded,k)

    autoencoder = Model(input_img, decoded)

    if verbose > 0:
        print(autoencoder.summary())

    return autoencoder




def build_model(image_dim, nlabels,nK, n_dil, kernel_size, drop_out, model_type, activation_hidden, activation_output, loss, verbose=0):
    if model_type == 'cnn-autoencoder':
       model = cnn_autoencoder(image_dim)
    elif model_type == 'cnn-binary-classifier':
       model = cnn_binary_classifier(image_dim)
    elif model_type == 'cnn_3D_classifier':
        model = cnn_3D_classifier(image_dim,nlabels)

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

        model.compile(loss=loss, optimizer=Adam(0.001))

    elif 'classifier' in model_type:

        model.compile(loss=loss, optimizer='rmsprop', metrics=[metric])


    print("Training size: {}\n Validation size: {}\n".format(X_train.shape, X_validate.shape))

    # train model
    # augmentation generator
    aug = ImageDataGenerator(rotation_range=1,
                             #width_shift_range=0.01,
                             #height_shift_range=0.01,
                             #zoom_range=0,
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


    # save model
    model.save(model_name)

    with open(history_fn, 'w+') as fp: json.dump(history.history, fp)

    return [model, history]

