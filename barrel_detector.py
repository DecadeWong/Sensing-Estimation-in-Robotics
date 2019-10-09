'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

def sigmoid(z):
        return 1 / ( 1 + np.exp(-z))
    
    
class BarrelDetector():
    def __init__(self):
        '''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
        self.weights = np.array([-190. , -452. ,  407.5,   -1. ])

    def segment_image(self, img):
        '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask =  np.zeros((img.shape[0], img.shape[1]))
        weights = self.weights
       #generating mask matrix
        for i in range(img.shape[0]):
           for j in range(img.shape[1]):
               elem = np.append(img_rgb[i,j,:],1)
               score = np.dot(weights, elem)
               labels = round(sigmoid(score))
               mask[i,j] = labels
        mask_img = np.asarray(mask)
        #plt.imshow(mask_img)
        #raise NotImplementedError
        return mask_img
    #def clean_noise(self, img):
        
    def get_bounding_box(self, img):
        '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
        		# YOUR CODE HERE
        #raise NotImplementedError
        mask_img = self.segment_image(img)
        #mask_img = mask_img.astype(int)
        #mask = mask_img.astype(np.uint8)
        #mask_blr = cv2.GaussianBlur(mask,(5,5),0)
        coords = []
        cdnts = []
        barrel_region = []
        boxes = []
        label_img = label(mask_img, connectivity = mask_img.ndim)
        props = regionprops(label_img)
        mask = np.uint8(label_img)
        areas = [a.area for a in props]
        areas.sort(reverse = True)
        for region in props:
            if region.area == areas[0] or (region.area == areas[1] and region.area > 5000):
                barrel_region.append(region)
                centroid = region.centroid
                coord = [int(centroid[1]), int(centroid[0])]
                coords.append(coord)
                cX,cY = coord[1],coord[0]
                cv2.circle(mask, (cX, cY), 5, (0, 0, 0), -1)
                    #cv2.putText(mask, cdnts, (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        #draw bounding boxes on original images
        for b in barrel_region:
             minr, minc, maxr, maxc = b.bbox
             bx = (minc, maxc, maxc, minc, minc)
             by = (minr, minr, maxr, maxr, minr)
             x1,y1 = bx[0], by[0]
             x2,y2 = bx[1], by[2]
             box  = [x1,y1,x2,y2]
             boxes.append(box)
        boxes = sorted(boxes, key = lambda x:x[0])
        print("boxes=", boxes)    
             #cv2.rectangle(img_rgb,(x1,y2),(x2,y1),(0,255,0),2)
        #M = cv2.moments(mask_blr)
        #cX = int(M["m10"]/M["m00"])
        #cY = int(M["m01"]/M["m00"])
        
        return boxes


if __name__ == '__main__':
    folder = "validationset"
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_box = img_rgb
        mask_img = my_detector.segment_image(img)
        boxes = my_detector.get_bounding_box(img)

        #boxes = my_detector.get_bounding_box(img)
        #cv2.imshow('mask_img',mask_img)
        fig = plt.figure()
        
        for box in boxes:
            [x1,y1,x2,y2]=box
            img_box = cv2.rectangle(img_box,(x1,y1),(x2,y2),(0,255,0),2)
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.subplot(221), plt.imshow(img,'gray'), plt.title('ORIGINAL')
        plt.subplot(222), plt.imshow(mask_img), plt.title('MASKED')
        #plt.subplot(223), plt.imshow(img_box), plt.title('BOXED')
        fig.suptitle(filename)
        plt.show(block=False)
        img_box = cv2.cvtColor(img_box,cv2.COLOR_BGR2RGB)
        cv2.imshow('figure', img_box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

