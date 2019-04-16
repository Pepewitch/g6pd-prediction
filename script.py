from crop import getCropRect, findCircle
from summary import getInsideCircle, getOutsideCircles
from os.path import join, dirname
from os import getcwd
from sys import argv
from skimage.io import imread
import pickle
import numpy as np

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
            inside['avg_g'],
            inside['avg_b'],
            outside['avg_r'],
            outside['avg_g'],
            outside['avg_b'],
        ])
    return output

def main():
    filepath = getFilepath()
    img = imread(filepath)
    data = getData(img)
    model = pickle.load(open(join(dirname(__file__),'model.p'), 'rb'))
    print(np.average(model.predict(data)))

if __name__ == '__main__':
    main()