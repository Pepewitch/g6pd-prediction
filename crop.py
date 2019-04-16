import cv2
import numpy as np

def getCropRect(sample):
    '''
    getCropRect(sample) -> rect
    
    @param sample, 3 channels numpy array image
    @return rect, 3 channels numpy array image which is cropped middle rectangle
    '''
    crop = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    crop[crop < 40] = 0
    crop[crop >= 40] = 255
    crop = cv2.morphologyEx(crop , cv2.MORPH_OPEN , cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (15,15)))
    _, contour , _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    pts1 = np.float32([polyTopleft,polyTopright,polyBottomleft,polyBottomright])
    pts2 = np.float32([[0,0],[700,0],[0,250],[700,250]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(sample,M,(700,250))
    return dst[25:225 , 50:650]

def findCircle(crop):
    '''Finding circle function
    findCircle(im) -> set
    
    @param crop, 3 channels numpy array image from getCropRect function
    @return set, a set of (x,y,r) from the input image
    '''
    hough_set = set()
    for index , rang in enumerate([(0,200) , (200,400) , (400,600)]):
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
