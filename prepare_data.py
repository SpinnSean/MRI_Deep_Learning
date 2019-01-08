import pandas as pd
import numpy as np
import nibabel as nib
from create_data_df import create_data_df
import os
import h5py
from helpers import nii2Numpy, normalize


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



def prepare_data(mainDir, data_dir, report_dir, input_str, ext, labelName, idColumn, covPath, imagesDfOut,ratios=[0.75,0.15]):

    data = {}

    # Setup file names and imagesDfput directories
    data["x_train_fn"] = os.path.join(mainDir,'x_train')
    data["y_train_fn"] = os.path.join(mainDir,'y_train')
    data["x_validate_fn"] = os.path.join(mainDir,'x_validate')
    data["y_validate_fn"] = os.path.join(mainDir,'y_validate')
    data["x_test_fn"] = os.path.join(mainDir,'x_test')
    data["y_test_fn"] = os.path.join(mainDir,'y_test')
    data["image_dim"] = [256,256]

    if os.path.exists(imagesDfOut):
        imagesDf = pd.read_csv(imagesDfOut)
        return [imagesDf, data]


    # Create the path, label, category data frame
    imagesDf = create_data_df(mainDir, data_dir, input_str, ext, labelName, covPath, idColumn)

    # fix to make sure enough test/train/validate
    attribute_category(imagesDf, 'train', 'labels', ratios[0])
    attribute_category(imagesDf, 'validate','labels', ratios[1])
    imagesDf.category.loc[ imagesDf.category == "unknown" ] = "test"

    # Create npy arrays with dataset split
    hdf5_path = os.path.join(mainDir, 'datasetSplit.hdf5')
    #create_hd5(imagesDf,data, hdf5_path)

    fname = os.path.join(mainDir, 'sMRI_{}.csv'.format(labelName))
    imagesDf.to_csv(fname, sep=',')

    return [imagesDf,data]


def create_hd5(imagesDf,data,hdf5_path):


    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    x_train = imagesDf['paths'][imagesDf.category=='train']
    y_train = imagesDf['labels'][imagesDf.category=='train']
    x_val = imagesDf['paths'][imagesDf.category=='validate']
    y_val = imagesDf['labels'][imagesDf.category=='validate']
    x_test = imagesDf['paths'][imagesDf.category=='test']
    y_test = imagesDf['labels'][imagesDf.category == 'test']


    train_shape = [x_train.shape[0]*256,256,256,1]
    val_shape = [x_val.shape[0]*256, 256, 256,1]
    test_shape = [x_test.shape[0]*256, 256, 256,1]

    # open hdf5 file and create arrays
    hdf5_f = h5py.File(hdf5_path, mode='w')
    hdf5_f.create_dataset("train_img", train_shape, dtype='float16')
    hdf5_f.create_dataset("validate_img", val_shape, dtype='float16')
    hdf5_f.create_dataset("test_img", test_shape, dtype='float16')

    hdf5_f.create_dataset("train_labels", [y_train.shape[0]*train_shape[1],1], np.int8)
    hdf5_f.create_dataset("validate_labels", [y_val.shape[0]*val_shape[1],1], np.int8)
    hdf5_f.create_dataset("test_labels", [y_test.shape[0]*test_shape[1],1], np.int8)

    total_index = {'train': 0,
                   'test': 0,
                   'validate':0}

    for index, row in imagesDf.iterrows():
        print('Index: {}'.format(index))
        if index % 10 == 0: print("Saving",imagesDf["category"][0],"images:",index, '/', imagesDf.shape[0] , end='\r')

        img3D = nii2Numpy(row.paths)
        label = row.labels
        img3D = normalize(img3D)
        img3D.reshape(list(img3D.shape) + [1])
        labels = np.full((img3D.shape[1], 1), label)

        for j in range(img3D.shape[1]):
            if img3D[:,j,:].sum() != 0:
                hdf5_f[row.category + "_img"][(total_index[row.category])] = img3D[:,j,:]
                hdf5_f[row.category + "_labels"][(total_index[row.category])] = labels[j]
                total_index[row.category] += 1

    np.save(data['x_train_fn'] + '.npy', hdf5_f['train_img'])
    np.save(data['x_validate_fn'] + '.npy',hdf5_f['validate_img'])
    np.save(data['x_test_fn'] + '.npy',hdf5_f['test_img'])

    np.save(data['y_train_fn'] +'.npy',hdf5_f['train_labels'])
    np.save(data['y_validate_fn'] +'.npy',hdf5_f['validate_labels'])
    np.save(data['y_test_fn'] +'.npy',hdf5_f['test_labels'])
    hdf5_f.close()
    print("")
    return 0




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
