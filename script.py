from crop import getCropRect, findCircle
from summary import getInsideCircle, getOutsideCircles
from os.path import join, dirname
from os import getcwd
from sys import argv
from cv2 import imread
# import pickle
import numpy as np
from joblib import load

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
    img = np.flip(imread(filepath), -1)
    data = getData(img)
    # model = pickle.load(open(join(dirname(__file__),'model.p'), 'rb'))
    model = load(join(dirname(__file__), 'model.joblib'))
    print(np.average(model.predict(data)))

if __name__ == '__main__':
    main()