
import matplotlib
matplotlib.use('Agg') # used for plotting from remote to local machine
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, subprocess

def formatImage(img):
    img = np.reshape(img,img.shape[1:-1]).astype(np.float32)
    return img


def sideBySide(img1,img2, grey=False):

    if grey:
        plt.subplot(121), plt.imshow(img1,cmap='gray', vmin = 0, vmax = 255), plt.title('Original')
        plt.subplot(122), plt.imshow(img2,cmap='gray', vmin = 0, vmax = 255), plt.title('Transformed')
        plt.show(block=True)

    else:
        plt.subplot(121), plt.imshow(img1), plt.title('Original')
        plt.subplot(122), plt.imshow(img2), plt.title('Transformed')
        plt.show(block=True)

def saveAndOpenPlot(image,imgDir,fname):
    fullPath = os.path.join(imgDir,fname)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    fig.savefig(fullPath)
    #subprocess.call(['xdg-open', fullPath])


def comparePredictions(test,pred,category,model_name):

    fname = os.path.join('.','processed/predict',category,model_name+'.pdf')
    fig, big_axes = plt.subplots(figsize=(32, 16), nrows=2, ncols=1, sharey=True)

    big_axes[0].set_title("Ground Truth", fontsize=22)
    big_axes[0].title.set_position([.5,0.95])
    big_axes[0].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_axes[0]._frameon = False


    big_axes[1].set_title("Predicted", fontsize=22)
    big_axes[1].title.set_position([.5, 0.95])
    big_axes[1].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    # removes the white frame
    big_axes[1]._frameon = False

    for i, j in zip(range(1, 6),range(6, 11)):
        ax1 = fig.add_subplot(2, 5, i)
        ax1.imshow(test[i,...,0].astype(np.float32), cmap='gray', aspect='equal')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(2, 5, j)
        ax2.imshow(pred[i,...,0].astype(np.float32), cmap='gray',aspect='equal')
        ax2.set_axis_off()

    fig.set_facecolor('w')
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.15, hspace=0)
    plt.savefig(fname)

def plotLoss(history,nb_epoch,model_name,show=False):
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(nb_epoch)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if show:
        plt.show()
    plt.savefig('./plots/' + model_name + '.pdf')

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