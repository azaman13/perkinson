from __future__ import division
import glob
import random
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib.colors as colors
from copy import deepcopy


def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)[1:][:, 3:]
    A = data[:3]
    # print 'A', A, A.shape
    Y = data[-1]
    return A, Y


def normalize_matrix(A, Y):

    nor_A = np.column_stack((A, Y))

    row_wise_norm = np.sqrt(np.sum(nor_A*nor_A, axis=1))

    for (i, norm) in enumerate(row_wise_norm):
        if norm != 0.00:
            A[i] = (1.0/norm)*A[i]

            Y[i] = (1.0/norm)*Y[i]

    return A, Y, row_wise_norm


def main():
    files = glob.glob("patient_wise_data/*.csv")
    for f in files:
        A, Y = read_data(f)
        A = A.transpose()
        # Now A looks like the following
        # [ V1   V2   V3 ]
        # ----------------
        # ...   ...   ... 

        A, Y, row_norm = normalize_matrix(A, Y)

        data = np.column_stack((A, Y))
        np.savetxt('normalized_data/'+f.split('/')[-1], data, delimiter=',')
        

if __name__ == '__main__':
    main()
