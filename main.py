import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resize not giving a shit about aspect ratio
def resize(img, w, h):
    return cv2.resize(img, (w, h))

# reshape the image to a 2D array of pixels and 3 color values (RGB)
def reshape(img):
    return np.float32(img.reshape((-1, 3)))

def k_means(img):
    # stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # k = 8, as in the article
    k = 8
    _, labels, (centers) = cv2.kmeans(reshape(img), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # return segmented image
    centers = np.uint8(centers)
    labels = labels.flatten()
    seg_img = centers[labels.flatten()]
    seg_img = seg_img.reshape(img.shape)
    return seg_img

images = os.listdir('dataset/')

for image in images:
    img = cv2.imread('dataset/'+image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, 512, 512)
    seg_image = k_means(img)
    #plt.imshow(seg_image)
    #plt.show()
    cv2.imwrite('results/'+image, seg_image)


