import cv2
import matplotlib.pyplot as plt
from helpers import nii2Numpy
from plotting import sideBySide
from nilearn import image
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


# def crop(images):
#
#     for i, img in enumerate(images[:,...,0]):
#         mask = img > 0
#         images[i,...,0] = img[np.ix_(mask.any(1), mask.any(0))]
#
#     return images


#
# def crop(imgMat):
#     nparts, nrow, ncol, _ = imgMat.shape
#     firstLitPix = False
#     boundingBox = []
#     [boundingBox.append({'yStart': [0,0],
#                     'yEnd': [0,0],
#                     'xStart': [0, 0],
#                     'xEnd': [0, 0],
#                     'Width': 0,
#                     'Height': 0}) for n in range(nparts)]
#     maxHeight = 0
#     maxWidth = 0
#     for p in range(nparts):
#
#         # yStart
#         for r in range(nrow):
#             s = imgMat[p,r,:,:].sum()
#             if s != 0:
#                 boundingBox[p]['yStart'] = r
#                 break
#         # yEnd
#         for r in reversed(range(nrow)):
#             s = imgMat[p,r,:,:].sum()
#             if s != 0:
#                 boundingBox[p]['yEnd'] = r
#                 break
#         # xStart
#         for c in range(ncol):
#             s = imgMat[p,:,:,c].sum()
#             if s != 0:
#                 boundingBox[p]['xStart'] = c
#                 break
#
#         # xEnd
#         for c in reversed(range(ncol)):
#             s = imgMat[p,:,:,c].sum()
#             if s != 0:
#                 boundingBox[p]['xEnd'] = c
#                 break
#
#         boundingBox[p]['Width'] = boundingBox[p]['xEnd'] - boundingBox[p]['xStart']
#         boundingBox[p]['Height'] = boundingBox[p]['yEnd'] - boundingBox[p]['yStart']
#
#     maxWidth, maxHeight = findMax(boundingBox)
#     xStart, xEnd, yStart, yEnd = findCoords(boundingBox,nparts)
#     sideLength = max(maxWidth, maxHeight)
#
#     newImgMat = np.empty((nparts, sideLength,sideLength, sideLength))
#     for p in range(nparts):
#         newImgMat[p,:,:,:] = imgMat[p,xStart:sideLength+xStart,:,yStart:sideLength+yStart]
#
#     return [newImgMat, [sideLength, sideLength]]
#

def get_crop_dims(img3D):
    maxDim=0
    for img2D in np.rollaxis(img3D,1):
        img2D = img2D.squeeze()
        mask = img2D > 0
        dims = list(img2D[np.ix_(mask.any(1), mask.any(0))].shape)
        if max(dims) > maxDim:
            maxDim = max(dims)

    return [maxDim,maxDim]


def cropDimensions(imagesDf):

    nsub = imagesDf.shape[0]
    # 1. First pass is to get all cropped dimensions
    cropDims=[]
    for index, row in imagesDf.iterrows():
        img3D = nii2Numpy(row.paths)
        numImg = [img3D.shape[0]]
        #numNonZero = [sum([1 for x in np.rollaxis(img3D,1) if x.sum() != 0])]
        cropDims.append(numImg + get_crop_dims(img3D))

    # 2. Find max values for bounding box
    maxNum = max(list(zip(*cropDims))[0])
    maxWidth = max(list(zip(*cropDims))[1])
    maxHeight = max(list(zip(*cropDims))[2])
    sideLength = max(maxWidth, maxHeight)

    return [maxNum, sideLength]


def crop(img):
    mask = img > 0
    return img[np.ix_(mask.any(1), mask.any(0))]



# def crop(imgMat):
#
#     nparts, nrow, ncol, ndepth = imgMat.shape
#     firstLitPix = False
#     boundingBox = []
#     [boundingBox.append({'yStart': [0,0],
#                     'yEnd': [0,0],
#                     'xStart': [0, 0],
#                     'xEnd': [0, 0],
#                     'Width': 0,
#                     'Height': 0}) for n in range(nparts)]
#
#     dims = []
#     for i, img in enumerate(imgMat[:, ..., 0]):
#         mask = img > 0
#         dims.append(list(img[np.ix_(mask.any(1), mask.any(0))].shape))
#
#
#
#
#
#     newImgMat = np.empty((nparts, sideLength, sideLength, sideLength))
#     for i, img in enumerate(imgMat[:, ..., 0]):
#         mask = img > 0
#         newImgMat[i,...,0] = img[np.ix_(mask.any(1), mask.any(0))]
#
#     return [newImgMat, [sideLength, sideLength]]



#
# def unitTest1():
#     path = '/Volumes/Storage/Work/Data/Neuroventure/sub-002/brainmask.nii'
#     img = nii2Numpy(path)[:,:,:,0]
#     imgRotated = rotate3D(img,60)
#
#     # x axis check
#     sideBySide(img[100,:,:], imgRotated[100,:,:], grey=True)
#
#     # y axis check
#     sideBySide(img[:,100,:], imgRotated[:,100,:], grey=True)
#
#     # z axis check
#     sideBySide(img[:,:,100], imgRotated[:,:,100], grey=True)

#unitTest1()