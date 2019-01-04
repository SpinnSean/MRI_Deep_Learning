import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def sideBySide(img1,img2, grey=False):

    if grey:
        plt.subplot(121), plt.imshow(img1,cmap='gray', vmin = 0, vmax = 255), plt.title('Original')
        plt.subplot(122), plt.imshow(img2,cmap='gray', vmin = 0, vmax = 255), plt.title('Transformed')
        plt.show(block=True)

    else:
        plt.subplot(121), plt.imshow(img1), plt.title('Original')
        plt.subplot(122), plt.imshow(img2), plt.title('Transformed')
        plt.show(block=True)



# def panelPNG(imgPaths):
#     images = [Image.open(str(p)).convert('L') for p in imgPaths]
#     imagesCopy = [im.copy() for im in images]
#     imagesList = [im for im in imagesCopy]
#     numImages = len(imagesList)
#     panel_size = np.sqrt(numImages)
#     #panel_size = np.floor(np.sqrt(numImages)) + 1
#
#     widths, heights = zip(*(i.size for i in imagesList))
#     total_width = sum(widths)
#     total_height = sum(heights)
#
#     imagesWidth, imagesHeight = images[0].size
#
#
#     panel = Image.new('L', (int(panel_size*max(widths)), int(panel_size*max(heights))))
#
#     x_offset = 0
#     y_offset = 0
#     i = 0
#
#     while :
#         for y in range(0, int(panel_size*max(heights)), max(heights)):
#             for x in range(0, int(panel_size*max(widths)), max(widths)):
#                 #print(left, top)
#                 im = Image.open(str(imgPaths[i]))
#                 print("Img size: {}".format(im.size))
#                 #panel.paste(im, (x_offset, y_offset))
#                 panel.paste(im, (y, x))
#
#                 # if (i>0) and (i % panel_size == 0):
#                 #     y_offset += images[i].size[1]
#                 #     x_offset = 0
#                 #
#                 #
#                 # else:
#                 #     x_offset += images[i].size[0]
#
#                 #if i == 470:
#                  #   l=i
#                 i+=1
#                 print(i)
#                 if i == len(imgPaths):
#
#
#
#     panel.save('/Volumes/Storage/Work/Data/Neuroventure/test.jpg')


def panelPNG(imgPaths):
    images = [Image.open(str(p)).convert('L') for p in imgPaths]
    imagesCopy = [im.copy() for im in images]
    imagesList = [im for im in imagesCopy]
    numImages = len(imagesList)
    #panel_size = np.sqrt(numImages)
    panel_size = np.floor(np.sqrt(numImages)) + 1

    widths, heights = zip(*(i.size for i in imagesList))
    total_width = sum(widths)
    total_height = sum(heights)

    imagesWidth, imagesHeight = images[0].size

    panel = Image.new('L', (int(panel_size * max(widths)), int(panel_size * max(heights))))

    x_offset = 0
    y_offset = 0
    i = 0

    for i in range(len(imgPaths)):
        im = Image.open(str(imgPaths[i]))
        panel.paste(im, (x_offset, y_offset))
        if (i > 0) and (i % panel_size == 0):
            y_offset += max(heights)
            x_offset = 0

        else:
            x_offset += max(widths)


    panel.save('/Volumes/Storage/Work/Data/Neuroventure/test.jpg')