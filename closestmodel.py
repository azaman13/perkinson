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

    for f in training_files:
        A, Y = read_data(f)
        # print A.transpose().shape
        # print Y.shape
        reg = linear_model.Ridge(alpha=.5)
        model = reg.fit(A.transpose(), Y)
        models[f.split('/')[-1]] = model
        # models.append((model, f.split('/')[-1]))

        error = LA.norm(A.transpose().dot(model.coef_) - Y)

    # This is the list of all predictions for every patient in the
    # testing set
    all_prediction_vectors = []

    # list of all true labels
    all_label_vectors = []

    # For testing
    for f in testing_files:
        A, Y = read_data(f)

        avg_pred_y = np.zeros(Y.shape)

        # take the test point and make it go through
        # all the models
        for (pat_no, model) in models.iteritems():
            pred_y = A.transpose().dot(model.coef_)

            error_i = LA.norm(pred_y - Y)

            avg_pred_y = np.add(avg_pred_y, pred_y)
            # print '->', avg_pred_y

            # print '\t Patient: ', pat_no, 'Error:', error_i
        avg_pred_y = 1.0 / (len(models)) * avg_pred_y

        # now for each patient we have the avg pred vector and true Y
        # let's save the pred vectors in a list and later use
        # numpy's column stack to create a matrix of predictions

        all_prediction_vectors.append(avg_pred_y)

        all_label_vectors.append(Y)

        print 'Done predicting: ', f.split('/')[-1]
        # print Y, Y.shape
        # print '---'
        # print avg_pred_y, avg_pred_y.shape
        # print '========================================'

    # Here each col is the pred vector for a patient: is a 100x42 matrix
    pred_matrix = np.column_stack(all_prediction_vectors)

    # here each col is the true label vector for a patient: is a 100x42 matrix
    label_matrix = np.column_stack(all_label_vectors)

    print 'pred matrix:', pred_matrix.shape
    print 'label matrix:', label_matrix.shape

    error_matrix = np.subtract(label_matrix, pred_matrix)
    print 'Error matrix:', error_matrix.shape
    print 'p', pred_matrix[0]
    print 'l', label_matrix[0]
    print 'e', error_matrix[0]
    error = LA.norm(error_matrix)
    print 'Final error:', error

    print '**************************************************************'

    # now for testing, a sample point goes through all the models we have.
    # we actually want to find which patient is most similar to our sample
    # input testing point. Find the most similar parient and use the model
    # that represent that patient inorder to do prediction. In this version
    # avoid the averaging effect

    # So go through each test data point and find the closest model from the
    # list of models using the training data point
    error_vectors = []
    test_errors = []
    for f in testing_files:
        print f
        testing_A, Y = read_data(f)
        closest_model, closest_patient = find_closest_model(
            models, training_files, testing_A)
        pred_y = A.transpose().dot(closest_model.coef_)

        error = LA.norm(Y - pred_y)
        test_errors.append(error)
        print f.split('/')[-1], ' patient is closest to:', closest_patient, 'pred_error', error
        error_vectors.append(Y - pred_y)

    err_matrix = np.column_stack(error_vectors).transpose()
    # im = plt.matshow(err_matrix, cmap=pl.cm.hot, norm=colors.LogNorm(
    #     vmin=0.0001, vmax=64), aspect='auto')

    # plt.colorbar(im)
    # plt.ylabel('Patient id')
    # plt.xlabel('feature number')
    # plt.title('Error Matrix heatmap for testing using Ridge Regression')
    # plt.show()

    # plot training errors
    fig2 = plt.figure()
    plt.plot(range(len(test_errors)), test_errors, '-b',
             label='Feature Prediction error for Test set(Ridge regression)', linewidth=2.0)
    plt.xlabel('Patient id')
    plt.ylabel('Error (l2 norm of the predicted features)')
    plt.title('Error for test set')
    plt.savefig('Ridge_test_error.png')
    # for training error


def find_closest_model(models, training_files, testing_A):
    best_patient = ''
    min_similarity = 9999

    # we go through each training sample point
    # and try to find the training data point
    # that is closest to the sample testing_A
    # and return the corresponding model that
    # should be used
    for f in training_files:
        A, Y = read_data(f)
        similarity = LA.norm(testing_A - A)
        if similarity < min_similarity:
            min_similarity = similarity
            best_patient = f.split('/')[-1]

    return models[best_patient], best_patient


if __name__ == '__main__':
    main()
