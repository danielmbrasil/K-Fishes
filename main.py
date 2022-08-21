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


def histogram(image, nbins):
    N, M = image.shape
    hist = np.zeros(nbins).astype(int)
    for x in range(N):
        for y in range(M):
            hist[image[x,y]] += 1
    return hist

def histogram_equalization(image, values):
    hist = histogram(image, values)

    histC = np.zeros(hist.shape).astype(int)

    histC[0] = hist[0]
    for i in range(1, values):
        histC[i] = hist[i] + histC[i-1]
  
    N, M = image.shape
    
    hist_transform = np.zeros(values).astype(np.uint8)

    image_eq = np.zeros(image.shape).astype(np.uint8)

    for r in range(values): 
        s = ((values-1)/float(M*N))*histC[r]
        image_eq[ np.where(image == r) ] = s
        hist_transform[r] = s
    
    return (image_eq, hist_transform)

def plot_histogram(hist, name='', xlabel='Intensity', ylabel='Frequency'):
    plt.bar(range(256), hist)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

images = os.listdir('dataset/')

for image in images:
    img = cv2.imread('dataset/'+image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize(img, 50, 50)
    seg_image = k_means(img)
    
    (B, G, R) = cv2.split(seg_image)
    gray_image = 0.2989 * R + 0.587 * G + 0.1141 * B # as defined in the article

    gray_image = cv2.equalizeHist(np.uint8(gray_image))
    cv2.imwrite('results/he'+image, gray_image)
    cv2.imwrite('results/seg'+image, seg_image)

    #img_eq, img_t = histogram_equalization(B, values=256)
    #histeq_img = histogram(img_eq, 256)
    #cv2.imwrite('teste/he_old'+image, img_eq)
    #cv2.imshow("",img_eq)
    
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(gray_image) + 30
    cv2.imwrite('results/final'+image, final_img)

    _, im_binary_th = cv2.threshold(final_img, 127, 255, cv2.THRESH_BINARY)   
    cv2.imwrite('results/bin'+image, im_binary_th)
    #plot_histogram(histeq_img, 'Equalized histogram ')
    #plt.imshow(seg_image)
    #plt.show()
    #cv2.imwrite('results/'+image, seg_image)


