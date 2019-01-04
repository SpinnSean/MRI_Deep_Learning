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

def findMax(boundingBox):
    maxWidth = boundingBox[0]['Width']
    maxHeight = boundingBox[0]['Height']
    for imgBox in boundingBox:
        if imgBox['Width'] > maxWidth:
            maxWidth = imgBox['Width']

        if imgBox['Height'] > maxHeight:
            maxHeigt = imgBox['Height']

    return maxWidth, maxHeight

def findCoords(boundingBox, nparts):
    xStart = 1000
    xEnd = 0
    yStart = 1000
    yEnd = 0
    for p in range(nparts):
        imgBox = boundingBox[p]
        if imgBox['xStart'] < xStart:
            xStart = imgBox['xStart']
        if imgBox['yStart'] < yStart:
            yStart = imgBox['yStart']
        if imgBox['xEnd'] > xEnd:
            xEnd = imgBox['xEnd']
        if imgBox['yEnd'] > yEnd:
            yEnd = imgBox['yEnd']
    return xStart, xEnd, yStart, yEnd



def crop(imgMat):
    nrow, ncol, ndepth, nparts = imgMat.shape
    firstLitPix = False
    boundingBox = []
    [boundingBox.append({'yStart': [0,0],
                    'yEnd': [0,0],
                    'xStart': [0, 0],
                    'xEnd': [0, 0],
                    'Width': 0,
                    'Height': 0}) for n in range(nparts)]
    maxHeight = 0
    maxWidth = 0
    for p in range(nparts):

        # yStart
        for r in range(nrow):
            s = imgMat[r,:,:,p].sum()
            if s != 0:
                boundingBox[p]['yStart'] = r
                break
        # yEnd
        for r in reversed(range(nrow)):
            s = imgMat[r,:,:,p].sum()
            if s != 0:
                boundingBox[p]['yEnd'] = r
                break
        # xStart
        for c in range(ncol):
            s = imgMat[:,:,c,p].sum()
            if s != 0:
                boundingBox[p]['xStart'] = c
                break

        # xEnd
        for c in reversed(range(ncol)):
            s = imgMat[:,:,c,p].sum()
            if s != 0:
                boundingBox[p]['xEnd'] = c
                break

        boundingBox[p]['Width'] = boundingBox[p]['xEnd'] - boundingBox[p]['xStart']
        boundingBox[p]['Height'] = boundingBox[p]['yEnd'] - boundingBox[p]['yStart']

    maxWidth, maxHeight = findMax(boundingBox)
    xStart, xEnd, yStart, yEnd = findCoords(boundingBox,nparts)
    sideLength = max(maxWidth, maxHeight)

    newImgMat = np.empty((sideLength,ndepth,sideLength,nparts))
    for p in range(nparts):
        newImgMat[:,:,:,p] = imgMat[xStart:sideLength+xStart,:,yStart:sideLength+yStart,p]

    return [newImgMat, [sideLength, sideLength]]











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