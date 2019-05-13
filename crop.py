import cv2
import numpy as np
import json
from os.path import dirname, join, realpath

config = json.load(open(join(dirname(realpath(__file__)), 'config.json')))
ratio = config['ratio']

if ratio[0] > ratio[1]:
    height = 700
    width = ratio[1] * 700 // ratio[0]
else:
    width = 700
    height = ratio[0] * 700 // ratio[1]

def findRect(sample):
    _, contour , _ = cv2.findContours(sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour) == 0:
        y, x = sample.shape
        return np.float32([[0,0],[x-1,0],[x-1,y-1],[0,y-1]])
    def getArea(attr):
        return attr[2] * attr[3]
    max_area_contour = max(contour , key=lambda c : getArea(cv2.boundingRect(c)))
    hull = cv2.convexHull(max_area_contour)
    poly = cv2.approxPolyDP(hull , 10 , True)
    x, y, w, h = cv2.boundingRect(poly)
    topleft = np.array([x,y])
    topright = np.array([x+w , y])
    bottomleft = np.array([x , y+h])
    bottomright = np.array([x+w , y+h])
    polyTopleft = min(poly, key=lambda i: np.sqrt(np.sum((topleft - i)**2)))
    polyTopright = min(poly, key=lambda i: np.sqrt(np.sum((topright - i)**2)))
    polyBottomleft = min(poly, key=lambda i: np.sqrt(np.sum((bottomleft - i)**2)))
    polyBottomright = min(poly, key=lambda i: np.sqrt(np.sum((bottomright - i)**2)))
    return np.float32([polyTopleft,polyTopright,polyBottomright,polyBottomleft])

def getCropRect(sample):
    '''
    getCropRect(sample) -> rect, coordinate
    
    @param sample, 3 channels numpy array image
    @return rect, 3 channels numpy array image which is cropped middle rectangle
    '''
    crop = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    crop[crop < 40] = 0
    crop[crop >= 40] = 255
    crop = cv2.morphologyEx(crop , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (15,15)))
    pts1 = findRect(crop)
    pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(sample,M,(width,height))
    return dst[25:height-25 , 50:width-50], pts1

def findCircle(crop):
    '''Finding circle function
    findCircle(im) -> set
    
    @param crop, 3 channels numpy array image from getCropRect function
    @return set, a set of (x,y,r) from the input image
    '''
    hough_set = set()
    circle_width = (width-100)//3
    for index , rang in enumerate([(0,circle_width) , (circle_width,circle_width*2) , (circle_width*2,circle_width*3)]):
        p1 = crop[:,rang[0]:rang[1]]
        hough = cv2.HoughCircles(cv2.cvtColor(p1 , cv2.COLOR_RGB2GRAY) , cv2.cv2.HOUGH_GRADIENT, 2, 400,
                          param1=25,
                          param2=15,
                          minRadius=40,
                          maxRadius=60)
        if hough is not None:
            for x , y , r in hough[0,:]:
                hough_set.add((x+rang[0],y,r))
    return hough_set
