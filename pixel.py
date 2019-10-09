import logging
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from roipoly import RoiPoly

train_set = "trainset"
valid_set = "validationset"
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)


def roicrop(img):

# Show the image
    fig = plt.figure()
    
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.title("left click: line segment         right click: next figure")
    plt.show(block=False)

# Let user draw first ROI
    roi = RoiPoly(color='r', fig=fig)



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask =  roi.get_mask(gray)
    return roi, mask

    
def end_mouse(event, x, y, flag, param):
    pass

blue_barrel = []
blue_other = []
not_blue = []


"""for filename in os.listdir(train_set):
    # read one image in training set
    img = cv2.imread(os.path.join(train_set, filename), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Crop Blue Barrels")
    roi, mask = roicrop(img_rgb)
    
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if mask[i,j] != 0:
                blue_barrel.append([img_rgb[i, j, 0], img_rgb[i, j, 1], img_rgb[i, j, 2]])
    
blue_barrel = np.array(blue_barrel, dtype=np.uint8)
np.save('blue_barrel.npy', blue_barrel)

for filename in os.listdir(train_set):
    # read one image in training set
    img = cv2.imread(os.path.join(train_set, filename), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Crop Blue Others")
    roi, mask = roicrop(img_rgb)
    
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if mask[i,j] != 0:
                blue_other.append([img_rgb[i, j, 0], img_rgb[i, j, 1], img_rgb[i, j, 2]])
                
blue_other = np.array(blue_other, dtype=np.uint8)
np.save('blue_other.npy', blue_other)"""

for filename in os.listdir(train_set):
    # read one image in training set
    img = cv2.imread(os.path.join(train_set, filename), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Crop Not Blues")
    roi, mask = roicrop(img_rgb)
    
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            if mask[i,j] != 0:
                not_blue.append([img_rgb[i, j, 0], img_rgb[i, j, 1], img_rgb[i, j, 2]])

      
not_blue = np.array(not_blue, dtype=np.uint8)
np.save('not_blue.npy', not_blue)
