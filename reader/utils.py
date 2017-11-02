__author__ = 'cipriancorneanu'

import numpy as np

def ml2onehot(ml, dim):
    ''' '''
    integer = ml.dot(2**np.arange(ml.size)[::-1])
    onehot = np.zeros((1, np.power(2,dim)))
    onehot[:,integer] = 1

    return onehot

def onehot2ml(onehot, dim):
    if np.sum(onehot)==0:
        return np.zeros((1,dim))
    else:
        num = np.where(onehot==1)[0][0]
        print(num)
        out = [int(x) for x in np.binary_repr(num, width=dim)]
        print out
        return out

def int2onehot(num,dim):
    out = np.zeros(dim)
    out[num]=1
    return out

def onehot2int(onehot):
    '''Transform one-hot label into integer'''
    if np.sum(onehot)>0:
        return np.where(onehot==1)[0][0]
    else:
        return 0

def ml2ul(input, coding):
    '''Transform multi-label input into uni-label one-hot space by specifying the coding'''
    n_classes = len(coding)+1
    output = [coding.index(list(x))+1 if list(x) in coding else 0 for x in input]

    return [int2onehot(x, n_classes) for x in output]