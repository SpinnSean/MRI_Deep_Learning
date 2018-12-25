import matplotlib.pyplot as plt

def sideBySide(img1,img2, grey=False):

    if grey:
        plt.subplot(121), plt.imshow(img1,cmap='gray', vmin = 0, vmax = 255), plt.title('Original')
        plt.subplot(122), plt.imshow(img2,cmap='gray', vmin = 0, vmax = 255), plt.title('Transformed')
        plt.show(block=True)

    else:
        plt.subplot(121), plt.imshow(img1), plt.title('Original')
        plt.subplot(122), plt.imshow(img2), plt.title('Transformed')
        plt.show(block=True)
