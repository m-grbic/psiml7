from PIL import Image
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

BORDER_SIZE = 10
IMG_HEIGHT = 256
IMG_WIDTH = 352
IMG_HEIGHT_RESCALE = 284
IMG_WIDTH_RESCALE = 392

def rescale_img(img, output_shape=(IMG_HEIGHT_RESCALE, IMG_WIDTH_RESCALE)):
    # Rescaled images
    images_rescaled = np.zeros((1, 3, output_shape[0], output_shape[1]))

    # Border size
    border = BORDER_SIZE

        # Rescale images
    images_rescaled[0,:,:,:] = np.swapaxes(np.swapaxes(resize(
        np.swapaxes(np.swapaxes(img[0,:,:,:],0,1),1,2),
        output_shape = (output_shape[0]+2*border, output_shape[1]+2*border, 3),
        clip=True, 
        anti_aliasing=True,
        #multichannel=True,
        preserve_range=True
        ),2,1),0,1)[:,border:-border,border:-border]

    # Save images
    return images_rescaled

def centerCrop(imgs):

    y1 = int( (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
    y2 = int( IMG_HEIGHT_RESCALE - (IMG_HEIGHT_RESCALE - IMG_HEIGHT) // 2 )
    x1 = int( (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )
    x2 = int( IMG_WIDTH_RESCALE - (IMG_WIDTH_RESCALE - IMG_WIDTH) // 2 )

    return imgs[:,:,y1:y2,x1:x2]

if __name__ == '__main__':
    # Import image
    img = np.zeros((1, 3, IMG_HEIGHT_RESCALE, IMG_WIDTH_RESCALE))
    img = Image.open("preprocessing/sample_image.jpg")

    # Initialize transformations
    img = np.array(img)
    img = img[np.newaxis, ...]
    img = np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    # Rescale
    img = rescale_img(img=img)

    # Crop
    img = centerCrop(img)

    # plt.imshow(img[0].astype('uint8'))
    # plt.show()