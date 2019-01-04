import nibabel as nib
import numpy as np
import os
import re
from PIL import Image
import pandas as pd
from glob import glob
from pathlib import Path

def getAllNums(s):
    nums = re.findall(r'\d+', s)[0]
    return nums


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

    images = [{'data': nib.load(p), 'path': p} for p in imagePaths]
    #print(images)
    dims = validateDims(images)
    numpyImages = np.zeros( (dims + (len(images),)) )

    for i, img in enumerate(images):
        numpyImages[:,:,:,i] = img['data'].get_fdata()

    return numpyImages

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


def load_data(mainDir, imagesDf, PNG_DIM, NUM_IMG):

    niiPaths = imagesDf.paths.tolist()
    allPNGs = np.empty((len(niiPaths), NUM_IMG, PNG_DIM[0], PNG_DIM[1]))

    for n, p in enumerate(niiPaths):
        subNum = getAllNums(p)
        path2PNG = glob(os.path.join(mainDir,'sub-'+subNum, 'png', '*.png'))
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

    return {'x_train': x_train,
            'x_val': x_val,
            'x_test': x_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test}




def unitTest1():
    path = '/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii'
    images = nii2Numpy(path)
    print(images.shape)



