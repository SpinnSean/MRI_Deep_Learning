import pandas as pd
import numpy as np
import nibabel as nib
from create_data_df import create_data_df
from keras.utils import to_categorical
import os
import h5py
from helpers import *
from image_transformations import *

def evaluate_class_distribution(imagesDf):
    categories = ['train','validate','test']
    grp = imagesDf.groupby(by=['labels', 'category'])
    ratios = {k:{} for k in categories}
    for c in categories:
        n1 = grp.get_group((0,c)).shape[0]
        n2 = grp.get_group((1, c)).shape[0]
        ratios[c] = [n1,n2]

    print(pd.DataFrame(ratios))


def attribute_category(imagesDf, category, labelName, ratio, verbose=1):
    ''' This function distributes each subject in a 'train' or 'test' category.
    Args:
        imagesDf (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject.
        ratios (list): a list containing the proportions of train/test
            subjects. should sum to 1 and supposedly it has been tested before.
    Returns:
        imagesDf (pd.DataFrame): a pd.DataFrame that contains the info of all files
            by subject where the 'category' column has been set to either
            train or test depending the result of the random draw.
            The value of test or train is the same for a given subject.
    '''
    nImages = imagesDf.shape[0]
    n = int(round(nImages * ratio))
    i = 0

    category_classes = pd.Series(imagesDf['labels'])
    unique_category_classes = category_classes.unique()


    while True:
        for r in unique_category_classes:
            unknown_imagesDf = imagesDf[(imagesDf.category == "unknown") & (category_classes == r)]
            n_unknown = unknown_imagesDf.shape[0]
            if n_unknown == 0: continue
            random_i = np.random.randint(0, n_unknown)
            row = imagesDf[imagesDf.category == "unknown"].iloc[random_i,]
            imagesDf.loc[imagesDf.index[imagesDf.subjects == row.subjects], 'category'] = category
            i += imagesDf.loc[imagesDf.index[imagesDf.subjects == row.subjects], 'category'].shape[0]
            if i >= n: break

        n_unknown = imagesDf[(imagesDf.category == "unknown") & (category_classes == r)].shape[0]
        if i >= n or n_unknown == 0: break

    if verbose > 0:
        print(category, ": expected/real ratio = %3.2f / %3.2f" % (
        100. * ratio, 100. * imagesDf.category.loc[imagesDf.category == category].shape[0] / nImages))


def count_valid_samples(imagesDf):
    
    total_slices = 0
    imagesDf['valid_samples'] = np.repeat(0, imagesDf.shape[0])
    imagesDf['total_samples'] = np.repeat(0, imagesDf.shape[0])
    for index, row in imagesDf.iterrows():
        subMRI = nii2Numpy(imagesDf.paths)

        subMRI = normalize(subMRI)

        imagesDf['total_samples'].iloc[index] = subMRI.shape[0]
        valid_slices = 0
        for j in range(subMRI.shape[0]):
            if subMRI[j].sum() != 0:
                valid_slices += 1
        imagesDf['valid_samples'].iloc[index] = valid_slices
        total_slices += valid_slices
    return (total_slices)


# TODO: fix hardcoded image dimensions in case of existing file.
def prepare_data(mainDir, data_dir, report_dir, input_str, ext, labelName, idColumn, covPath, imagesDfOut,ratios=[0.75,0.15]):

    data = {}

    # Setup file names and imagesDfput directories
    data["x_train_fn"] = os.path.join(mainDir,'x_train')
    data["y_train_fn"] = os.path.join(mainDir,'y_train')
    data["x_validate_fn"] = os.path.join(mainDir,'x_validate')
    data["y_validate_fn"] = os.path.join(mainDir,'y_validate')
    data["x_test_fn"] = os.path.join(mainDir,'x_test')
    data["y_test_fn"] = os.path.join(mainDir,'y_test')


   # if os.path.exists(imagesDfOut):
    #    imagesDf = pd.read_csv(imagesDfOut)
     #   return [imagesDf, data]


    # Create the path, label, category data frame
    imagesDf = create_data_df(mainDir, data_dir, input_str, ext, labelName, covPath, idColumn)

    # fix to make sure enough test/train/validate
    attribute_category(imagesDf, 'train', 'labels', ratios[0])
    attribute_category(imagesDf, 'validate','labels', ratios[1])
    imagesDf.category.loc[ imagesDf.category == "unknown" ] = "test"

    evaluate_class_distribution(imagesDf)

    # Create npy arrays with dataset split
    hdf5_path = os.path.join(mainDir, 'datasetSplit.hdf5')
    if not os.path.exists(hdf5_path):
        data["image_dim"] = create_hd5(imagesDf,data,hdf5_path)
    else:
        data["image_dim"] = [256,256,256]


    fname = os.path.join(mainDir, 'sMRI_{}.csv'.format(labelName))
    imagesDf.to_csv(fname, sep=',')

    return [imagesDf,data]

# TODO: Fix the labelling. Not same size as number of images!
# TODO: Ignoring zero sum slices
def create_hd5(imagesDf,data,hdf5_path,cropping=False):


    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    x_train = imagesDf['paths'][imagesDf.category=='train']
    y_train = imagesDf['labels'][imagesDf.category=='train']
    x_val = imagesDf['paths'][imagesDf.category=='validate']
    y_val = imagesDf['labels'][imagesDf.category=='validate']
    x_test = imagesDf['paths'][imagesDf.category=='test']
    y_test = imagesDf['labels'][imagesDf.category == 'test']

    #[numImages, sideLength] = cropDimensions(imagesDf)
    numImages = 256
    sideLength = 256
    while sideLength % 2 != 0:
        sideLength+=1

    data["image_dim"] = [sideLength, sideLength, sideLength]

    train_shape = [x_train.shape[0]*numImages,sideLength,sideLength,1]
    val_shape = [x_val.shape[0]*numImages, sideLength, sideLength,1]
    test_shape = [x_test.shape[0]*numImages, sideLength, sideLength,1]

    # open hdf5 file and create arrays
    hdf5_f = h5py.File(hdf5_path, mode='w')
    hdf5_f.create_dataset("train_img", train_shape, dtype='float16')
    hdf5_f.create_dataset("validate_img", val_shape, dtype='float16')
    hdf5_f.create_dataset("test_img", test_shape, dtype='float16')

    hdf5_f.create_dataset("train_labels", [y_train.shape[0]*numImages,1], np.int8)
    hdf5_f.create_dataset("validate_labels", [y_val.shape[0]*numImages,1], np.int8)
    hdf5_f.create_dataset("test_labels", [y_test.shape[0]*numImages,1], np.int8)

    total_index = {'train': 0,
                   'test': 0,
                   'validate':0}


    for index, row in imagesDf.iterrows():
        print('Preparing data for subject #{}'.format(index))
        #if index % 10 == 0: print("Saving",imagesDf["category"][0],"images:",index, '/', imagesDf.shape[0])

        img3D = nii2Numpy(row.paths)
        label = row.labels
        img3D = normalize(img3D)
        img3D.reshape(list(img3D.shape) + [1])
        #labels = np.full((img3D.shape[1], 1), label)
        #labels = np.full(img3D.shape[1], label)

        # for j, img in enumerate(np.rollaxis(img3D,1)):
        #     if img.sum() != 0: # do not ignore 0 sum slices
        #         if cropping:
        #             img = crop(img)
        #             offset1 = sideLength - img.shape[0]
        #             offset2 = sideLength - img.shape[1]
        #             img= np.pad(img,((0,offset1),(0, offset2)), "constant")
        #
        #     img = np.reshape(img,(list(img.shape) + [1]))

        hdf5_f[row.category + "_img"][(total_index[row.category])] = img3D
        hdf5_f[row.category + "_labels"][(total_index[row.category])] = label
        total_index[row.category] += 1

    np.save(data['x_train_fn'] + '.npy', hdf5_f['train_img'])
    np.save(data['x_validate_fn'] + '.npy',hdf5_f['validate_img'])
    np.save(data['x_test_fn'] + '.npy',hdf5_f['test_img'])

    np.save(data['y_train_fn'] +'.npy',hdf5_f['train_labels'])
    np.save(data['y_validate_fn'] +'.npy',hdf5_f['validate_labels'])
    np.save(data['y_test_fn'] +'.npy',hdf5_f['test_labels'])
    hdf5_f.close()
    print("")

    return [sideLength,sideLength,sideLength]



def dataConfiguration(X_train, X_validate,X_test, Y_train, Y_validate, Y_test, model_type,imgDim,nlabels):

    if model_type == 'cnn-autoencoder':
        # This model takes the 50 middle slices of every subject 2D image

        #X_train = X_train[extractMostInfSlice(X_train)]
        #X_validate = X_validate[extractMostInfSlice(X_validate)]

        X_train = X_train[extractMiddleSlices(X_train, imgDim[0])]
        X_validate = X_validate[extractMiddleSlices(X_validate, imgDim[0])]

        X_train = X_train.astype('float32')
        X_validate = X_validate.astype('float32')
        X_test = X_test.astype('float32')

        # shuffle the data
        X_train, _ = quickShuffle(X_train)
        X_validate, _ = quickShuffle(X_validate)

    if model_type == 'cnn-binary-classifier':
        Y_train = to_categorical(Y_train, num_classes=nlabels)
        Y_validate = to_categorical(Y_validate, num_classes=nlabels)

        X_train, Y_train = shufflePair(X_train, Y_train)
        X_validate, Y_validate = shufflePair(X_validate, Y_validate)
        X_test, Y_test = shufflePair(X_test, Y_test)

        X_train = X_train.astype('float32')
        X_validate = X_validate.astype('float32')
        X_test = X_test.astype('float32')

    if model_type == 'cnn_3D_classifier':
        Y_train = to_categorical(Y_train, num_classes=nlabels)
        Y_validate = to_categorical(Y_validate, num_classes=nlabels)

        X_train, Y_train = shufflePair(X_train, Y_train)
        X_validate, Y_validate = shufflePair(X_validate, Y_validate)
        X_test, Y_test = shufflePair(X_test, Y_test)

        X_train = X_train.astype('float32')
        X_validate = X_validate.astype('float32')
        X_test = X_test.astype('float32')


    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test



def unitTest1():

    mainDir = 'maindir/'

    # Create fake dataframe as it should be done by create_data_imagesDf
    n=50
    subjects=[]
    paths=[]
    labels=[]
    i=1

    while(i<n):
        if np.random.random() > 0.5:
           labels.append('Male')
        else:
            labels.append('Female')

        subjects.append('sub-{:03d}'.format(i))
        paths.append(os.path.join(mainDir, 'sub-{:03d}'.format(i), 'sub-{:03d}_T1w.mnc'.format(i)))
        i+=1

    imagesDf=pd.DataFrame({'subjects': subjects,
                     'paths': paths,
                     'labels': labels})

    imagesDf["category"] = "unknown"

    return imagesDf
