import cv2
import numpy as np
from config import ratio
import os
import math

def validateCircle(image):
    hough = cv2.HoughCircles(image, 
                             cv2.cv2.HOUGH_GRADIENT, 
                             2, 
                             400, 
                             param1=25, 
                             param2=15, 
                             minRadius=math.floor(image.shape[0] * 0.4), 
                             maxRadius=math.ceil(image.shape[0] * 0.6))
    if hough is not None:
        min_length = 0.4 * image.shape[0]
        max_length = 0.6 * image.shape[0]
        for x , y , r in hough[0,:]:
            if r > min_length and r < max_length and x > min_length and x < max_length and y > min_length and y < max_length:
                return True
    return False

def findCircles(image, circularity_threshold = 0.95, r_weight = 0.33, g_weight = 0.33, b_weight = 0.33):
    '''
    image is bgr image
    '''
    # image = cv2.bitwise_not(image)
    short_size = min(image.shape[0], image.shape[1])
#     im = (image[:,:,2] // (1/r_weight) + image[:,:,1] // (1/g_weight) + image[:,:,0] // (1/b_weight)).astype(np.uint8)
    gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.medianBlur(gray_im, min(math.ceil(short_size / 300), 3))
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(im)
    circles = []

    for i in range(1, num_labels):
        left, top, width, height, area = stats[i]
        center_x, center_y = centroids[i]
        circularity = max(width, height) / min(width, height)
        r = min(width, height) // 2
        if circularity > circularity_threshold and r > math.ceil(short_size/100) and validateCircle(gray_im[top: top+height, left: left+width]):
            circles.append([center_x, center_y, r])
    # for x, y, r in circles:
    #     cv2.circle(im, (int(x),int(y)) , int(r) , (0,0,255) , 2)
    # cv2.imshow('thresh', im)
    return circles