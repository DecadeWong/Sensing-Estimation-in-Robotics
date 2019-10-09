# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:57:53 2019

@author: dronelab
"""

import os, cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
filename = "38.png"
img = cv2.imread(os.path.join("trainset",filename))
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_box = img_rgb
mask_img = my_detector.segment_image(img)
boxes = my_detector.get_bounding_box(img)

        #boxes = my_detector.get_bounding_box(img)
        #cv2.imshow('mask_img',mask_img)
fig = plt.figure()

for box in boxes:
    [x1,y1,x2,y2]=box
    img_box = cv2.rectangle(img_box,(x1,y2),(x2,y1),(0,255,0),2)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221), plt.imshow(img,'gray'), plt.title('ORIGINAL')
plt.subplot(222), plt.imshow(mask_img), plt.title('MASKED')
cv2.imwrite(filename+'_mask.png',mask_img)
#plt.subplot(223), plt.imshow(img_box), plt.title('BOXED')
cv2.imwrite(filename+'_boxed.png',img_box)
fig.suptitle(filename)
plt.show(block=False)
img_box = cv2.cvtColor(img_box,cv2.COLOR_BGR2RGB)
cv2.imshow('figure', img_box)
cv2.waitKey(0)
cv2.destroyAllWindows()