import numpy as np

coef = np.array([-0.21741104,  0.24249119, -0.01685429,  0.1394792 , -0.24525309, 0.08908307])
intercept = -7.605807281061505

def predict(data):
    if(type(data) != np.ndarray):
        data = np.array(data)
    return data.dot(coef) + intercept