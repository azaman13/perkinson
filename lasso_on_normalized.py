from __future__ import division
import glob
import random
import numpy as np
from sklearn import linear_model
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib.colors as colors
import seaborn as sns

def read_data(filename):
    f = open(filename)
    f.readline()
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    A = data[:, :3]
    Y = data[:, -1]
    return A, Y


def main():
    random.seed(5)
    # list of models for each patient
    models = {}
    tolerance = 0.2
    files = glob.glob("normalized_data/*.csv")

    random.shuffle(files)

    training_files = files[:100]
    testing_files = files[100:]

    D_matrix = np.zeros((100, 300))
    Y_matrix = np.zeros((100, 100))
    print 'Reading Training Data'
    i = 0
    for f in training_files:
        A, Y = read_data(f)
        A = A.transpose()

        new_row = np.concatenate((A[0], A[1], A[2]), axis=0)
        D_matrix[i] = new_row
        Y_matrix[i] = Y

        i += 1

    # Now training
    # print 'Y===>'
    # print Y_matrix
    # print 'Y shape:', Y_matrix.shape, 'D shape:', D_matrix.shape
    # print np.count_nonzero(Y_matrix), np.count_nonzero(D_matrix)
    reg = linear_model.Lasso(alpha=.01, max_iter=10000)
    model = reg.fit(D_matrix, Y_matrix)
    # print model.coef_
    # print 'model dim', model.coef_.shape

    num_non_zero_features = np.count_nonzero(model.coef_)
    print 'number of non zero:', num_non_zero_features
    #print D_matrix.dot(model.coef_.transpose())
    # print D_matrix.dot(model.coef_.transpose()).shape
    # #
    # print '---------------'
    # print D_matrix.dot(model.coef_.transpose()) - Y_matrix
    W = model.coef_.transpose()
    #
    test_D_matrix = np.zeros((100, 300))  # data matrix
    i = 0
    print 'Loading Testing Data'

    # Do training error
    train_errors = []
    train_error_vectors = []
    for f in training_files:
        A, Y = read_data(f)
        A = A.transpose()
        train_row = np.concatenate((A[0], A[1], A[2]), axis=0)

        train_pred_Y = train_row.dot(W)
        train_error_vector = train_pred_Y - Y
        train_error = LA.norm(train_error_vector)

        train_errors.append(train_error)
        train_error_vectors.append(train_error_vector)

    print 'Running Testing...'
    # now testing
    test_errors = []
    test_error_vectors = []
    er_matrix = np.zeros((42, 100))

    for f in testing_files:
        test_A, test_Y = read_data(f)
        test_A = test_A.transpose()

        test_row = np.concatenate((test_A[0], test_A[1], test_A[2]), axis=0)

        pred_y = test_row.dot(W)

        # print test_Y, test_Y.shape

        # now compute the error matrix using pred_Y_matrix and test_Y_matrix
        # for each patient
        er_vector = pred_y - test_Y
        er_matrix[i] = er_vector
        # er_vectors.append(list(er_vector))
        error = LA.norm(er_vector)
        test_errors.append(error)
        test_error_vectors.append(er_vector)
        i += 1
    print 'Doing feature accuracy calculations'
    # We want to find how many of the elements (i.e feature scores) of the vector of test_error_vectors
    # are within the 'tolerance' variable. Features that are inside the tolerance range
    # are correctly predicted and note which of these features are they
    feature_acuracy_per_patient = []

    for e_v in test_error_vectors:
        matched_features = []
        not_matched_features = []
        for i in range(len(e_v)):
            feature_num = i + 1
            # feature score is within the tolerance
            if np.fabs(e_v[i]) < tolerance:
                matched_features.append(feature_num)
            else:
                not_matched_features.append(feature_num)
        accuracy = (100.0 * len(matched_features))/( len(matched_features) + len(not_matched_features))
        #print 'Num features predicted correctly=', accuracy

        feature_acuracy_per_patient.append(accuracy)
    print 'Plotting feature prediction accuracy graph'
    fig1 = plt.figure()
    plt.plot(range(len(feature_acuracy_per_patient)), feature_acuracy_per_patient, '-r', linewidth=1.0, ls='-')

    plt.ylabel("Percentage correctly predicted features")
    plt.xlabel("Patient Number")
    plt.title(
        "LASSO(normalized): Percentage of features predicted correctly for patients from Test set")
    fig1.savefig('lasso_model_figs/normalized/Lasso_normalized_feature_accuracy.png')

    fig2 = plt.figure(figsize=(12,16))
    # Error Matrix Plotting
    print 'Plotting Heatmap for error matrix'
    # The Error Matrix Heat map
    ax = sns.heatmap(np.fabs(er_matrix), cmap="YlOrRd", xticklabels=False)
    ax.set(xlabel='feature number', ylabel='patient id', title='Error Matix on Test set')
    a = ax.get_figure()
    a.savefig('lasso_model_figs/normalized/lasso_normalized_test_error_matrix.png')

    print 'Plotting feature error distributions'
    # plots the feature error distribution
    for feature_num in range(42):
        # for all 42 patients
        fig3 = plt.figure()
        feature_i = er_matrix[feature_num, :]
        plt.hist(feature_i)
        plt.title("Lasso (normalized) Histogram with 'auto' bins for feature #" +
                  str(feature_num + 1))
        name = 'lasso_model_figs/normalized/feature_distribution_plots/feature_' + str(feature_num + 1) + '_histogram'
        fig3.savefig(name, bbox_inches='tight')

    # plot the errors
    fig4 = plt.figure()
    plt.plot(range(len(test_errors)), test_errors, '-b',
             label='Feature Prediction error for Test set', linewidth=2.0)
    plt.plot(range(len(train_errors)), train_errors, '-g',
             label='Feature Prediction error for Train set', linewidth=2.0)
    plt.ylabel("Error = l2_norm( pred_feature_vector - true_feature_vector)")
    plt.xlabel("Patient Number")
    plt.title(
        "LASSO(normalized): Feature Prediction error for Test patients")
    plt.legend(loc='upper right')
    # plt.show()
    fig4.savefig('lasso_model_figs/normalized/Lasso_Feature_pred_error.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
