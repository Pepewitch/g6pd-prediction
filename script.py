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
import argparse

selected_method = 3
isRed = False

def getFilepath(filename):
    if filename[0] == '/':
        return filename
    else:
        return join(getcwd() , filename)

def getFeature(inside, outside, method):
    if method == 1:
        return [
            inside['avg_r'],
            inside['avg_g'],
            inside['avg_b'],
            inside['std_r'],
            inside['std_g'],
            inside['std_b'],
            outside['avg_r'],
            outside['avg_g'],
            outside['avg_b'],
            outside['std_r'],
            outside['std_g'],
            outside['std_b']
        ]
    elif method == 2:
        return [
            inside['avg_r'],
            inside['avg_g'],
            inside['avg_b'],
            outside['avg_r'],
            outside['avg_g'],
            outside['avg_b'],
        ]
    else:
        return [
            inside['avg_r'] - outside['avg_r'],
            inside['avg_g'] - outside['avg_g'],
            inside['avg_b'] - outside['avg_b'],
        ]

def getData(crop, circles, method=selected_method):
    outside = getOutsideCircles(crop, circles)
    output = []
    for circle in circles:
        inside = getInsideCircle(crop, circle)
        output.append(getFeature(inside, outside, method))
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
    ac = 0
    if len(data) > 0:
        ac = np.average(model.predict(data, selected_method, isRed))
    cv2.polylines(bgrFrame, np.array([coordinate], dtype=np.int32), isClosed=True, color=(0,255,0), thickness=3)
    drawCircles(cropRect, circles)
    drawText(cropRect, 'Ac/Hb: {}'.format(math.floor(ac * 1000)/1000))
    bgr = cv2.cvtColor(cropRect, cv2.COLOR_RGB2BGR)
    return bgr

def main():
    parser = argparse.ArgumentParser(description="G6PD Prediction script working with a real-time camera or an image file.\nDefault is using the camera.")
    parser.add_argument('-r', '--red', action='store_true', help='Applying red light.')
    parser.add_argument('-c', '--pi-camera', action='store_true', help='Specific camera to be pi-camera.')
    parser.add_argument('-i', '--image', type=str, help='Image path to be predicted.')
    parser.add_argument('-m', '--method', type=int, help='Preprocessing method.\n1 is using avg and std of both background and foreground.\n2 is using avg of both background and foreground.\n3 is using avg of foreground - avg of background.')
    args = parser.parse_args(argv[1:])
    isRed = args.red
    piCamera = args.pi_camera
    if args.method is not None:
        selected_method = args.method
    if args.image is not None:
        filepath = getFilepath(args.image)
        img = np.flip(imread(filepath), -1)
        crop, _ = getCropRect(img)
        circles = findCircle(crop)
        data = getData(crop, circles)
        print(np.average(model.predict(data, selected_method, isRed)))
    else:
        openCamera(processFrame,piCamera)

if __name__ == '__main__':
    main()