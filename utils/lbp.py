import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def load_lbp(img):
    img_gray=img
    # img=img.permute(2,1,0)
    # img_gray =  cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_lbp = np.zeros((256, 256), np.uint8)
    for i in range(0, 256):
        for j in range(0,256):
            img_lbp[i, j,] = lbp_calculated_pixel(img_gray, i, j)
    # img_lbp=rgb2gray(rgb2gray)

    return img_lbp

def image_to_edge(image, sigma):

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    #calculate the LBP map
    edge = image_to_tensor()(Image.fromarray(load_lbp(gray_image)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))
    return edge, gray_image

