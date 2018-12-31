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



def panelPNG(imgPaths):
    images = [Image.open(str(p)).convert('L') for p in imgPaths]
    imagesCopy = [im.copy() for im in images]
    imagesList = [im for im in imagesCopy]
    numImages = len(imagesList)
    panel_size = np.sqrt(numImages)

    widths, heights = zip(*(i.size for i in imagesList))
    total_width = sum(widths)
    total_height = sum(heights)

    imagesWidth, imagesHeight = images[0].size


    panel = Image.new('L', (int(panel_size*widths[0]), int(panel_size*heights[0])))

    x_offset = 0
    y_offset = 0
    i = 0

    for left in range(0, int(panel_size*widths[0]), widths[0]):
        for top in range(0, int(panel_size*heights[0]), heights[0]):
            print(left, top)
            im = Image.open(str(imgPaths[i]))
            panel.paste(im, (x_offset, y_offset))

            if (i>0) and (i % panel_size == 0):
                y_offset += images[i].size[1]
                x_offset = 0


            else:
                x_offset += images[i].size[0]

            i+=1



    # for im in images:
    #     print("y_offset: {}\n x_offset: {}".format(y_offset, x_offset))
    #     panel.paste(im, (x_offset, y_offset))
    #
    #     if i / 100 == 1:
    #         print("100th")
    #
    #     if (i>0) and (i % (panel_size - 1) == 0):
    #         y_offset += im.size[1]
    #         x_offset = 0
    #
    #
    #     else:
    #         x_offset += im.size[0]
    #
    #     i+=1

    panel.save('/Volumes/Storage/Work/Data/Neuroventure/test.jpg')
