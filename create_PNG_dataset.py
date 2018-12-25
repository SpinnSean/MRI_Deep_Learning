import os
from PIL import Image
from pathlib import Path
import numpy as np
from helpers import nii2Numpy, updateSubjectDf, normalize
from image_transformations import *


def safely_remove_file(path):
    try:
        path.unlink()
    except OSError as e:  ## if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))


def create_png(img, pathOut):
    """
    This function takes a 3D numpy array and saves all 2D slices as
    PNGs.

    :param image: numpy array of a 2D image
    :param path: path to save the image
    """
    PIL_img = Image.fromarray(img).convert('RGB')
    if not pathOut.parent.exists(): pathOut.parent.mkdir(parents=True, exist_ok=True)

    PIL_img.save(pathOut)

def create_PNG_dataset(imagePaths, augmentation=True, max_theta=60):

    imgArrays = nii2Numpy(imagePaths)

    if augmentation:
        for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePaths):
            path = Path(path)
            print("Processing {}".format(path.parents[0].parts[-1]))
            augmentedImg = rotate3D(subjectImage, max_theta)
            augmentedPath =  Path(str(path.parents[0]) + 'a')
            updateSubjectDf(augmentedImg,augmentedPath)


    else:
        for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePaths):
            path = Path(path)
            print("Processing {}".format(path.parents[0].parts[-1]))
            k=1
            for slice2D in np.rollaxis(subjectImage,2):
                pathOut = path.parent / 'png' / 'slice_{:03d}.png'.format(k)

                # if picture already exists, remove it first
                if pathOut.exists(): safely_remove_file(pathOut)

                slice2D = normalize(slice2D)
                # if sum of pixels is 0, skip slice
                if slice2D.sum() == 0:
                    continue

                create_png(slice2D,pathOut)
                print("Creating 2D slice number: {:03d}".format(k))
                k+=1



def unitTest1():
    imagePaths = ['/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii',
                  '/Volumes/Storage/Work/Data/Neuroventure/sub-003/brainmask.nii']
    create_PNG_dataset(imagePaths)
    print("unitTest1 is finished.")

unitTest1()