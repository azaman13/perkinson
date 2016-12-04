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
    print 'starting to train...'
    
    train_errors = []
    
    for f in training_files:
        A, Y = read_data(f)
        # print A.transpose().shape
        # print Y.shape
        reg = linear_model.Ridge(alpha=.5)
        model = reg.fit(A.transpose(), Y)
        models[f.split('/')[-1]] = model
        # models.append((model, f.split('/')[-1]))

        error = LA.norm(A.transpose().dot(model.coef_) - Y)
        train_errors.append(error)

    # This is the list of all predictions for every patient in the
    # testing set
    all_prediction_vectors = []

    # list of all true labels
    all_label_vectors = []

    test_errors = []

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
        avg_error = LA.norm(avg_pred_y - Y)
        test_errors.append(avg_error)


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

    # print 'pred matrix:', pred_matrix.shape
    # print 'label matrix:', label_matrix.shape

    error_matrix = np.subtract(label_matrix, pred_matrix)
    # print 'Error matrix:', error_matrix.shape
    # print 'p', pred_matrix[0]
    # print 'l', label_matrix[0]
    # print 'e', error_matrix[0]
    error = LA.norm(error_matrix)
    print 'Final error:', error



    error_fig = plt.figure()
    plt.plot(range(len(test_errors)), test_errors, '-r',
             label='Feature Prediction error for Test set', linewidth=1.0, ls='-')
    plt.plot(range(len(train_errors)), train_errors, '-g',
             label='Feature Prediction error for Train set', linewidth=1.0)
    plt.ylabel("Error = l2_norm( pred_feature_vector - true_feature_vector)")
    plt.xlabel("Patient Number")
    plt.title(
        "Feature Prediction error for patients using AVG Model")
    plt.legend(loc='upper right')
    plt.show()





if __name__ == '__main__':
    main()