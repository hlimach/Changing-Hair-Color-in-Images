from ast import operator
import numpy as np

def get_distance(input, target):
    input_rgb = input[1:-1]
    target_rgb = target[1:-1]
    dist = np.sum(np.power(np.subtract(input_rgb, target_rgb),2))
    return dist

def get_transforms(params):
    print('gave transforms')
    return 0