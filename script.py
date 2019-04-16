from crop import getCropRect, findCircle
from summary import getInsideCircle, getOutsideCircles
from os.path import join
from os import getcwd
from sys import argv
from skimage.io import imread

def getFilepath():
    if(len(argv) == 1):
        raise Exception('Please input filename')
    filename = argv[1]
    if filename[0] == '/':
        return filename
    else:
        return join(getcwd() , filename)

def getData(img):
    crop = getCropRect(img)
    circles = findCircle(crop)
    outside = getOutsideCircles(crop, circles)
    output = []
    for circle in circles:
        inside = getInsideCircle(crop, circle)
        output.append([
            inside['avg_r'],
            inside['std_r'],
            inside['avg_g'],
            inside['std_g'],
            inside['avg_b'],
            inside['std_b'],
        ])

def main():
    filepath = getFilepath()
    img = imread(filepath)
    
if __name__ == '__main__':
    main()