import os
import cv2
import numpy as np

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
    img = resize(img, 128, 128)
    img_b = img
    cv2.imwrite('results/res-' + image, img_b)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_image = k_means(img)
    
    # split color channels and convert image to grayscale
    (B, G, R) = cv2.split(seg_image)
    gray_image = 0.2989 * R + 0.587 * G + 0.1141 * B # as defined in the article

    # Histogram equalization
    gray_image = cv2.equalizeHist(np.uint8(gray_image))
    #cv2.imwrite('results/he'+image, gray_image)
    #cv2.imwrite('results/seg'+image, seg_image)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    gray_image = clahe.apply(gray_image)
    #cv2.imwrite('results/final'+image, gray_image)
    
    # convert image to binary (black and white)
    im_binary_th = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]
    
    # morphology process - opening
    final_img = cv2.erode(im_binary_th,np.ones((5,5), np.uint8),iterations = 1)
    final_img = cv2.dilate(final_img,np.ones((5,5), np.uint8),iterations = 1)
    morpho = cv2.morphologyEx(im_binary_th, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # closing
    morpho = cv2.morphologyEx(morpho, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    # edge detection using Canny
    edges = cv2.Canny(image=morpho, threshold1=100, threshold2=200)
#    cv2.imwrite('results/bin-' + image, morpho)
    cv2.imwrite('results/edge-' + image, edges)
    
    # draw edge around detected objects in original image
    contours = cv2.findContours(edges, 
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_b, contours[0], -1, (0,0,255), thickness = 2)

    # final result
    cv2.imwrite('results/result-' + image, img_b)

    # binary image
    #cv2.imwrite('results/bin-' + image, im_binary_th)

    # k-means segmented image
    #cv2.imwrite('results/' + image, seg_image)

