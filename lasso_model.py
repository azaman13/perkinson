import glob
import random
import numpy as np
from sklearn import linear_model
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib.colors as colors


def read_data(filename):
    f = open(filename)
    f.readline()
    org_data = np.genfromtxt(filename, delimiter=',')[1:]

    data = np.delete(org_data, [0, 1, 2], axis=1)
    A = data[: -1]
    Y = data[-1]
    return A, Y


def main():
    random.seed(5)
    # list of models for each patient
    models = {}

    files = glob.glob(
        "patient_wise_data/*.csv")

    random.shuffle(files)

    training_files = files[:100]
    testing_files = files[100:]

    D_matrix = np.zeros((100, 300))
    Y_matrix = np.zeros((100, 100))
    print D_matrix.shape
    i = 0
    for f in training_files:
        A, Y = read_data(f)
        new_row = np.concatenate((A[0], A[1], A[2]), axis=0)

        D_matrix[i] = new_row
        Y_matrix[i] = Y

        i += 1

    # Now training
    reg = linear_model.Lasso(alpha=.1, max_iter=10000)
    model = reg.fit(D_matrix, Y_matrix)
    num_non_zero_features = np.count_nonzero(model.coef_)
    print num_non_zero_features
    print model.coef_.shape
    print Y_matrix
    print '==============='
    print D_matrix.dot(model.coef_.transpose())

    print '---------------'
    print D_matrix.dot(model.coef_.transpose()) - Y_matrix
    W = model.coef_.transpose()

    test_D_matrix = np.zeros((100, 300))  # data matrix
    i = 0
    print '==============='

    # Do training error
    train_errors = []
    for f in training_files:
        A, Y = read_data(f)
        train_row = np.concatenate((A[0], A[1], A[2]), axis=0)
        train_pred_Y = train_row.dot(W)
        train_error = LA.norm(train_pred_Y - Y)
        train_errors.append(train_error)

    # now testing
    test_errors = []
    er_matrix = np.zeros((42, 100))
    for f in testing_files:
        test_A, test_Y = read_data(f)
        test_row = np.concatenate((test_A[0], test_A[1], test_A[2]), axis=0)
        # print test_row, test_row.shape
        # print W, W.shape

        pred_Y_matrix = test_row.dot(W)

        # print test_Y, test_Y.shape

        # now compute the error matrix using pred_Y_matrix and test_Y_matrix
        # for each patient
        er_vector = pred_Y_matrix - test_Y
        er_matrix[i] = er_vector
        # er_vectors.append(list(er_vector))
        error = LA.norm(er_vector)
        test_errors.append(error)
        i += 1
    print er_matrix.shape

    # plotting error matrix heatmap
    im = plt.matshow(er_matrix, cmap=pl.cm.hot, norm=colors.LogNorm(
        vmin=0.0001, vmax=64), aspect='auto')
    # pl is pylab imported a pl
    plt.colorbar(im)
    plt.ylabel('Patient id')
    plt.xlabel('feature number')
    plt.title('Error Matrix heatmap for testing')
    plt.show()

    for feature_num in range(100):
        # for all 42 patients
        fig = plt.figure()

        feature_i = er_matrix[:, feature_num]
        print len(feature_i)
        plt.hist(feature_i, bins='auto')
        plt.title("Histogram with 'auto' bins for feature #" +
                  str(feature_num + 1))
        name = 'feature_' + str(feature_num + 1) + '_histogram'
        fig.savefig(name, bbox_inches='tight')

    # plot the errors
    fig2 = plt.figure()
    plt.plot(range(len(test_errors)), test_errors, '-b',
             label='Feature Prediction error for Test set', linewidth=2.0)
    plt.plot(range(len(train_errors)), train_errors, '-g',
             label='Feature Prediction error for Train set', linewidth=2.0)
    plt.ylabel("Error = l2_norm( pred_feature_vector - true_feature_vector)")
    plt.xlabel("Patient Number")
    plt.title(
        "Feature Prediction error for patients using Sparse Learning (LASSO")
    plt.legend(loc='upper right')
    # plt.show()
    fig2.savefig('Lasso_Feature_pred_error.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
