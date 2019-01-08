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

    batch_size=25
    N = len(imagePaths)
    prev=0
    numImgs = []
    for b,k in enumerate(range(batch_size,N,batch_size)):
        print("PNG dataset creation: batch {} out of {}.".format(b,batch_size))
        imagePathsBatch = imagePaths[prev:k]
        imgArrays = nii2Numpy(imagePathsBatch)

        if augmentation:
            for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePathsBatch):
                path = Path(path)
                print("Processing {}".format(path.parents[0].parts[-2]))
                augmentedImg = rotate3D(subjectImage, max_theta)
                augmentedPath =  Path(str(path.parents[0]) + 'a')
               # updateSubjectDf(augmentedImg,augmentedPath)

        #imgArrays = nii2Numpy(imagePaths)
        [imgArrays, PNG_DIM] = crop(imgArrays)
        PNG_DIM=[256,256]

        for subjectImage, path in zip(np.rollaxis(imgArrays,3), imagePathsBatch):
            path = Path(path)
            print("Processing {}".format(path.parents[0].parts[-2]))
            #btRatioVec = brainTissueRatioVec(subjectImage)
            subPaths = []
            for j in axes:
                kk = 0
                for slice2D in np.rollaxis(subjectImage,j):
                    pathOut = path.parent / 'png' / 'slice_{:03d}_p{}.png'.format(kk,j)

                    # if picture already exists, remove it firsts
                    if pathOut.exists(): safely_remove_file(pathOut)

                    #slice2D = normalize(slice2D)
                    # if sum of pixels is 0, skip slice
                    #if (slice2D.sum() == 0) or (k % 2 == 0):
                     #   k += 1
                     #   continue

                    create_png(slice2D,pathOut)
                    subPaths.append(pathOut)
                    print("Creating 2D slice number: {:03d}".format(kk))
                    kk+=1

                numImgs.append(kk)
            #panelPNG(subPaths)
        prev=k

    NUM_IMG = validateNumImg(numImgs)

    return NUM_IMG, PNG_DIM





def unitTest1():
    imagePaths = ['/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii',
                  '/Volumes/Storage/Work/Data/Neuroventure/sub-003/brainmask.nii']
    create_PNG_dataset(imagePaths,augmentation=False)
    print("unitTest1 is finished.")

#unitTest1()