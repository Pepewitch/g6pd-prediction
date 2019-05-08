import numpy as np

weight = {
    'red': {
        1: {
            'coef': np.array([-0.10233776,  0.14363647,  0.04812154,  0.01586341, -0.02811235,-0.03841939,  0.16996753, -0.11154178, -0.01630679,  0.07126346,-0.31210494,  0.10059287]),
            'intercept': -9.569197981850884
            },
        2: {
            'coef': np.array([-0.10485498,  0.12506428,  0.06020227,  0.16539364, -0.135549  ,0.01069023]),
            'intercept': -14.20532299821739
        },
        3:{
            'coef': np.array([-0.08310911,  0.19461797,  0.00323115]),
            'intercept': 7.340878275690651
        }
    },
    'normal': {
        1: {
            'coef': np.array([-2.03801701e-01,  2.20818016e-01, -1.44153777e-02,  1.99054831e-01,9.05433117e-01, -6.72979009e-01,  6.98915975e-02, -1.70462925e-01,3.00825615e-04, -9.11066915e-01,  6.16036109e-01, -3.64288907e-01]),
            'intercept': 8.959067512157144
        },
        2: {
            'coef': np.array([-0.21741104,  0.24249119, -0.01685429,  0.1394792 , -0.24525309, 0.08908307]),
            'intercept': -7.605807281061505
        },
        3: {
            'coef': np.array([-0.22582901,  0.25996106, -0.02821643]),
            'intercept': 6.435703027492024
        }
    }
}

def predict(data, method, isRed):
    coef = weight['red' if isRed else 'normal'][method]['coef']
    intercept = weight['red' if isRed else 'normal'][method]['intercept']
    if(type(data) != np.ndarray):
        data = np.array(data)
    return data.dot(coef) + intercept