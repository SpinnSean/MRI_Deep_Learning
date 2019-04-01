import nibabel as nib
import numpy as np
import os
import ast
import re
import h5py
from PIL import Image
import pandas as pd
from glob import glob
from pathlib import Path
import numpy as np
from keras import backend as K
from functools import reduce

def readParameters(path2Args):
    args = pd.read_csv('mri_keras_args', header=None, sep=';')
    args_dict = dict(zip(args[0], args[1]))

    # Transform the type for some variables
    nonStrTypes = ['ratios',
                   'createPNGDataset',
                   'batch_size',
                   'nb_epoch',
                   'images_to_predict',
                   'clobber', 'n_dil',
                   'kernel_size',
                   'drop_out',
                   'pad_base',
                   'verbose',
                   'make_model_only',
                   'nGPU']
    for arg in nonStrTypes:
        args_dict[arg] = ast.literal_eval(args_dict[arg])

    return args_dict


# TODO: fix getAllNums to get all numbers including leading zeros
def getAllNums(s):
    #nums = re.findall(r'\d+', s)[0]
    nums = re.findall(r'\d+', s)[1]
    return nums


def quickShuffle(A):
    """
    Shuffles along the first dimention of an nD array

    :param A: ndarray
    :return: shuffled ndarray
    """
    perm = np.arange(A.shape[0])
    np.random.shuffle(perm)
    A = A[perm]
    return A,perm

def shufflePair(A,B):
    A, shuffled_ind =quickShuffle(A)
    B = B[shuffled_ind]
    return A,B

def validateDims(images):
    """
    Verifies that all images have the same dimensions
    and then returns them.

    :param images:
    :return:
    """
    dims = images[0]['data'].shape
    for img in images:
        if img['data'].shape != dims:
            print('Problem with subject {}'.format(Path(img['path']).parents[0].parts[-1]))
    return dims

def validateNumImg(numImgs):

    if any([n != numImgs[0] for n in numImgs]):
        print("Not all subjects have the same number of images.")
        return False
    else:
        return numImgs[0]

def nii2Numpy(imagePaths):

    if type(imagePaths) != list:
        imagePaths = [imagePaths]
        if len(imagePaths) == 0:
            print("No paths for nii2Numpy to process.")
            return -1

    #images = [{'data': nib.load(p), 'path': p} for p in imagePaths]
    images = nib.load(imagePaths[0])
    #print(images)
    #dims = validateDims(images)
    #numpyImages = np.zeros( (dims + (len(images),)) )
    #numpyImages = np.zeros((dims))

    #total = len(images)
    #for i, img in enumerate(images):
    #    #print("Loading images: {}% complete.".format(100*round(i/total,2)))
     #   numpyImages[...] = img['data'].get_fdata()

    return images.get_fdata()

def updateSubjectDf(newImg,path):

    dfPath = glob(os.path.join(str(path.parents[0]), 'sMRI_*'))
    imagesDf = pd.read_csv(dfPath)
    subjName = path.parts[-1]
    subjData = {'subjects': subjName,
                'paths': path,
                'labels': imagesDf.loc[imagesDf.subjects == subjName[:-1], ['labels']],
                'category': imagesDf.loc[imagesDf.subjects == subjName[:-1], ['category']]
                }
    imagesDf = imagesDf.append(subjData, ignore_index=True)
    imagesDf.to_csv(dfPath)

def normalize(A):
    '''
    performs a simple normalization from 0 to 1 of a numpy array. checks that the image is not a uniform value first
    args
        A -- numpy array
    returns
        numpy array (either A or normalized version of A)
    '''
    std_factor=1
    if np.std(A) > 0 : std_factor=np.std(A)
    A = (A - np.mean(A)) / std_factor

    scale_factor=np.max(A) - A.min()
    if  scale_factor == 0: scale_factor = 1
    A = (A - A.min()) /scale_factor
    return A


def brainTissueRatioVec(vol3D,axis=1):

    btRatioList = []
    for slice2D in np.rollaxis(vol3D, axis):
        slice2D_bin = np.where(slice2D > 0, 1, 0)
        btRatioList.append(slice2D_bin.sum())

    return btRatioList

def saveDataset(mainDir,dataDict):
    fname=os.path.join(mainDir,'trainTestValDataset.hdf5')
    if os.path.exists(fname):
        print("Saved dataset already exists. Loading from file {}...".format(fname))
        return loadDataFromFile(mainDir)

    f = h5py.File(fname, 'a')
    grp = f.create_group('data')
    for k,v in dataDict.items():
        print('Saving {}...'.format(k))
        grp.create_dataset(k, data=v)
    f.close()
    print("Dataset saved at {}".format(fname))

def loadDataFromFile(mainDir):
    fname = os.path.join(mainDir,'trainTestValDataset.hdf5')
    if not os.path.exists(fname):
        print("Saved dataset does not exist. Run mri_keras again, with the createDataset parameter set to true.")
        raise SystemExit
    f = h5py.File(fname,'r')
    dataDict = {'x_train': f['data/x_train']}

    f.close()



def load_data(mainDir, data_dir, imagesDf, PNG_DIM, NUM_IMG, fromFile=False):

    if fromFile:
        dataDict = loadDataFromFile(mainDir)

    else:

        niiPaths = imagesDf.paths.tolist()
        allPNGs = np.empty((len(niiPaths), NUM_IMG, PNG_DIM[0], PNG_DIM[1]))

        total=len(niiPaths)
        for n, p in enumerate(niiPaths):
            print("Loading PNGs into arrays: {}% completed.".format(100*round(n/total,2)))
            subNum = getAllNums(p)
            path2PNG = glob(os.path.join(mainDir,'sub-'+subNum, data_dir, 'png', '*.png'))
            path2PNG.sort()

            for i, png in enumerate(path2PNG):
                try:
                    img = Image.open(png).convert('L')
                    allPNGs[n,i,:,:] = np.array(img)

                except Exception as e:
                    print('Problem loading sub-{} PNGs.\n Path: {}\n Img size: {}\nError: {}'.format(subNum,png,img.size,e))

        train_ind = imagesDf.index[imagesDf.category == 'train']
        test_ind = imagesDf.index[imagesDf.category == 'test']
        val_ind = imagesDf.index[imagesDf.category == 'validate']

        if len(train_ind) + len(test_ind) + len(val_ind) != allPNGs.shape[0]:
            print("Problem with train/test/val split.")
            return -1

        x_train = allPNGs[train_ind,:,:,:]
        x_val = allPNGs[val_ind,:,:,:]
        x_test = allPNGs[test_ind,:,:,:]

        y_train = imagesDf.labels[train_ind].values
        y_val = imagesDf.labels[val_ind].values
        y_test = imagesDf.labels[test_ind].values

        dataDict = {'x_train': x_train,
                    'x_val': x_val,
                    'x_test': x_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test}

        saveDataset(mainDir,dataDict)

    return dataDict


def subsetImages(images, axis=1, start=0, end=None):

    if end is None:
        end = images.shape[-1]

    if axis == 0:
        A = images[:,start:end,:,:]
    elif axis == 1:
        A = images[:,:,start:end,:]
    elif axis == 2:
        A = images[:,:,:,start:end]

    return  A

def set_model_name(filename, target_dir, ext='.hdf5'):
    '''function to set default model name'''
    return  os.path.join(target_dir, os.path.splitext(os.path.basename(filename))[0]+ext)


def shuffled_index(A,B, dims, random_state):
    if A.shape[dims[0]] != B.shape[dims[1]]:
        exit(0)

    p = np.random.permutation(range(A.shape[dims[0]]))

    return p


def extractMostInfSlice(A,dim=256):

    prev = 0
    indices=[]
    for x in range(dim-1, A.shape[0], dim):
        subImage = A[prev:x,:,:]
        maxSum = -np.inf
        for ind,slice in enumerate(np.rollaxis(subImage,0)):
            if slice.sum() >= maxSum:
                maxSum = slice.sum()
                mostInfSlice = ind

        indices.append(prev+mostInfSlice)
        prev = x

    return indices

# TODO: fix the hardcoding of middle slices
def extractMiddleSlices(A,dim=256):

    # the numbers 115 and 190 are approximate bounds for
    # the middle of the brain on a 256 resolution
    #L = int(np.floor(115/256 * (dim-1)))
    #H = int(np.floor(190/256 * (dim-1)))
    L = 107
    H = 147

    prev = 0
    indices=[]
    for x in range(dim, A.shape[0], dim):
        lowInd = prev + L
        highInd = prev + H
        indices.extend(list(range(lowInd,highInd)))
        prev=x

    return indices


def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

from keras.layers import Lambda, concatenate
from keras import Model
import tensorflow as tf

def multi_gpu_model(model, gpus):
  if isinstance(gpus, (list, tuple)):
    num_gpus = len(gpus)
    target_gpu_ids = gpus
  else:
    num_gpus = gpus
    target_gpu_ids = range(num_gpus)

  def get_slice(data, i, parts):
    shape = tf.shape(data)
    batch_size = shape[:1]
    input_shape = shape[1:]
    step = batch_size // parts
    if i == num_gpus - 1:
      size = batch_size - step * i
    else:
      size = step
    size = tf.concat([size, input_shape], axis=0)
    stride = tf.concat([step, input_shape * 0], axis=0)
    start = stride * i
    return tf.slice(data, start, size)

  all_outputs = []
  for i in range(len(model.outputs)):
    all_outputs.append([])

  # Place a copy of the model on each GPU,
  # each getting a slice of the inputs.
  for i, gpu_id in enumerate(target_gpu_ids):
    with tf.device('/gpu:%d' % gpu_id):
      with tf.name_scope('replica_%d' % gpu_id):
        inputs = []
        # Retrieve a slice of the input.
        for x in model.inputs:
          input_shape = tuple(x.get_shape().as_list())[1:]
          slice_i = Lambda(get_slice,
                           output_shape=input_shape,
                           arguments={'i': i,
                                      'parts': num_gpus})(x)
          inputs.append(slice_i)

        # Apply model on slice
        # (creating a model replica on the target device).
        outputs = model(inputs)
        if not isinstance(outputs, list):
          outputs = [outputs]

        # Save the outputs for merging back together later.
        for o in range(len(outputs)):
          all_outputs[o].append(outputs[o])

  # Merge outputs on CPU.
  with tf.device('/cpu:0'):
    merged = []
    for name, outputs in zip(model.output_names, all_outputs):
      merged.append(concatenate(outputs,
                                axis=0, name=name))
    return Model(model.inputs, merged)




