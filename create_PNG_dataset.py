import os
from PIL import Image
from pathlib import Path
import numpy as np
from helpers import *
from plotting import panelPNG
from image_transformations import *


def safely_remove_file(path):
    try:
        path.unlink()
    except OSError as e:  ## if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))


def create_png(img, pathOut):
    """

    :param image: numpy array of a 2D image
    :param path: path to save the image
    """
    PIL_img = Image.fromarray(img).convert('RGB')
    if not pathOut.parent.exists(): pathOut.parent.mkdir(parents=True, exist_ok=True)

    PIL_img.save(pathOut)

def create_PNG_dataset(imagePaths, augmentation=False, max_theta=60, axes=[1]):

    imgArrays = nii2Numpy(imagePaths)

    if augmentation:
        for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePaths):
            path = Path(path)
            print("Processing {}".format(path.parents[0].parts[-1]))
            augmentedImg = rotate3D(subjectImage, max_theta)
            augmentedPath =  Path(str(path.parents[0]) + 'a')
           # updateSubjectDf(augmentedImg,augmentedPath)

    imgArrays = nii2Numpy(imagePaths)
    #[imgArrays, PNG_DIM] = crop(imgArrays)
    PNG_DIM=[256,256]

    for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePaths):
        path = Path(path)
        print("Processing {}".format(path.parents[0].parts[-1]))
        btRatioVec = brainTissueRatioVec(subjectImage)
        subPaths = []
        numImgs = []
        for j in axes:
            k = 0
            for slice2D in np.rollaxis(subjectImage,j):
                pathOut = path.parent / 'png' / 'slice_{:03d}_p{}.png'.format(k,j)

                # if picture already exists, remove it firsts
                if pathOut.exists(): safely_remove_file(pathOut)

                #slice2D = normalize(slice2D)
                # if sum of pixels is 0, skip slice
                #if (slice2D.sum() == 0) or (k % 2 == 0):
                 #   k += 1
                 #   continue

                create_png(slice2D,pathOut)
                subPaths.append(pathOut)
                print("Creating 2D slice number: {:03d}".format(k))
                k+=1

            numImgs.append(k)

    NUM_IMG = validateNumImg(numImgs)

    return NUM_IMG, PNG_DIM


        #panelPNG(subPaths)


def unitTest1():
    imagePaths = ['/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii',
                  '/Volumes/Storage/Work/Data/Neuroventure/sub-003/brainmask.nii']
    create_PNG_dataset(imagePaths,augmentation=False)
    print("unitTest1 is finished.")

#unitTest1()