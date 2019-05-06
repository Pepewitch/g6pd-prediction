from crop import getCropRect, findCircle
from summary import getInsideCircle, getOutsideCircles
from os.path import join, dirname
from os import getcwd
from sys import argv
from cv2 import imread
import cv2
import numpy as np
import model
from camera import openCamera
import math

def getFilepath():
    filename = argv[1]
    if filename[0] == '/':
        return filename
    else:
        return join(getcwd() , filename)

def getData(crop, circles):
    outside = getOutsideCircles(crop, circles)
    output = []
    for circle in circles:
        inside = getInsideCircle(crop, circle)
        output.append([
            inside['avg_r'],
            inside['avg_g'],
            inside['avg_b'],
            outside['avg_r'],
            outside['avg_g'],
            outside['avg_b'],
        ])
    return output

def drawCircles(img, circles):
    for x ,y ,r in circles:
        cv2.circle(img , (int(x),int(y)) , int(r) , (0,0,255) , 2)

def drawText(img, text):
    cv2.putText(img,text, (10,img.shape[0]-10), cv2.FONT_HERSHEY_COMPLEX, 1.2, 255, 2)

def processFrame(bgrFrame):
    rgb = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)
    cropRect, coordinate = getCropRect(rgb)
    circles = findCircle(cropRect)
    data = getData(cropRect, circles)
    ac = np.average(model.predict(data))
    cv2.polylines(bgrFrame, np.array([coordinate], dtype=np.int32), isClosed=True, color=(0,255,0), thickness=3)
    drawCircles(cropRect, circles)
    drawText(cropRect, 'Ac/HB: {}'.format(math.floor(ac * 1000)/1000))
    bgr = cv2.cvtColor(cropRect, cv2.COLOR_RGB2BGR)
    return bgr

def main():
    if len(argv) > 1:
        filepath = getFilepath()
        img = np.flip(imread(filepath), -1)
        crop, _ = getCropRect(img)
        circles = findCircle(crop)
        data = getData(crop, circles)
        print(np.average(model.predict(data)))
    else:
        openCamera(processFrame)

if __name__ == '__main__':
    main()