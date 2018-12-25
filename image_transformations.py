import cv2
import matplotlib.pyplot as plt
from helpers import nii2Numpy
from plotting import sideBySide
import numpy as np
import scipy.ndimage



def rotate(img,angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def rotate3D(img,max_theta):

    theta = np.random.uniform(-max_theta, max_theta)

    # rotate along x axis
    img2 = scipy.ndimage.interpolation.rotate(img, theta, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y axis
    img3 = scipy.ndimage.interpolation.rotate(img2, theta, mode='nearest', axes=(0, 2), reshape=False)

    # rotate along z axis
    imgRotated = scipy.ndimage.interpolation.rotate(img3, theta, mode='nearest', axes=(1, 2), reshape=False)

    return imgRotated


def unitTest1():
    path = '/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii'
    img = nii2Numpy(path)[:,:,:,0]
    imgRotated = rotate3D(img,60)

    # x axis check
    sideBySide(img[100,:,:], imgRotated[100,:,:], grey=True)

    # y axis check
    sideBySide(img[:,100,:], imgRotated[:,100,:], grey=True)

    # z axis check
    sideBySide(img[:,:,100], imgRotated[:,:,100], grey=True)

#unitTest1()