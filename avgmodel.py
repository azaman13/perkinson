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
    org_data = np.genfromtxt(filename, delimiter=',')[1:]

    data = np.delete(org_data, [0, 1, 2], axis=1)
    A = data[: -1]
    Y = data[-1]
    return A, Y


def main():
    random.seed(5)
    # list of models for each patient
    models = {}

    # Prediction and Testing feature match tolerance
    tolerance = 0.1

    files = glob.glob(
        "patient_wise_data/*.csv")

    random.shuffle(files)

    training_files = files[:100]
    testing_files = files[100:]
    print 'Starting to train...'

    train_errors = []

    for f in training_files:
        A, Y = read_data(f)
        # print A.transpose().shape
        # print Y.shape
        reg = linear_model.Ridge(alpha=.5)
        model = reg.fit(A.transpose(), Y)
        models[f.split('/')[-1]] = model

        error = LA.norm(A.transpose().dot(model.coef_) - Y)
        train_errors.append(error)

    # This is the list of all predictions for every patient in the
    # testing set
    all_prediction_vectors = []

    # list of all true labels
    all_label_vectors = []

    test_errors = []

    test_percent_accurately_predicted_features = []

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
        avg_error_vector = avg_pred_y - Y

        avg_error = LA.norm(avg_error_vector)

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

        print 'Doing feature accuracy calculations'
        # We want to find how many of the elements (i.e feature scores) of the avg_error_vector
        # are within the 'tolerance' variable. Features that are inside the tolerance range
        # are correctly predicted and note which of these features are they
        matched_features = []
        not_matched_features = []
        for i in range(len(avg_error_vector)):
            feature_num = i + 1
            # feature score is within the tolerance
            if avg_error_vector[i]< tolerance:
                matched_features.append(feature_num)
            else:
                not_matched_features.append(feature_num)

        print 'Num features predicted correctly=', len(matched_features), 'and INCORRECTLY=', len(not_matched_features)
        correctly_predicted_features = (100.0*len(matched_features))/len(avg_error_vector)
        print 'Percentance of features correctly predicted = ', correctly_predicted_features
        test_percent_accurately_predicted_features.append(correctly_predicted_features)

    print 'Plotting Feature accuracy'
    # now we plot the test_percent_accurately_predicted_features for testing as well
    fig1 = plt.figure()
    plt.plot(range(len(test_percent_accurately_predicted_features)), test_percent_accurately_predicted_features, '-r', linewidth=1.0, ls='-')

    plt.ylabel("Percentage correctly predicted features")
    plt.xlabel("Patient Number")
    plt.title(
        "AVG Model: Percentage of features predicted correctly for patients from Test set")
    fig1.savefig('avg_model_figs/not_normalized/avg_model_feature_accuracy.png')


    print 'Calculating Error Matrix'
    # Here each col is the pred vector for a patient: is a 100x42 matrix
    pred_matrix = np.column_stack(all_prediction_vectors)

    # here each col is the true label vector for a patient: is a 100x42 matrix
    label_matrix = np.column_stack(all_label_vectors)

    error_matrix = np.subtract(label_matrix, pred_matrix)

    error = LA.norm(error_matrix)

    error_matrix = error_matrix.transpose()
    print error_matrix
    print error_matrix.shape

    # print 'Plotting feature error distributions'
    # for feature_num in range(42):
    #     # for all 42 patients
    #     fig = plt.figure()
    #
    #     feature_i = error_matrix[feature_num, :]
    #     plt.hist(feature_i)
    #     plt.title("Avg Model: Histogram with 'auto' bins for feature #" +
    #               str(feature_num + 1))
    #     name = 'avg_model_figs/not_normalized/feature_distribution_plots/feature_' + str(feature_num + 1) + '_histogram'
    #     fig.savefig(name, bbox_inches='tight')


    fig2 = plt.figure(figsize=(12,16))
    print 'Creating heatman for Error matrix'
    ax = sns.heatmap(np.fabs(error_matrix), cmap="YlOrRd", xticklabels=False)
    ax.set(xlabel='feature number', ylabel='patient id', title='AVG Model Error Matix on Test set')
    a = ax.get_figure()
    a.savefig('avg_model_figs/not_normalized/avg_model_error_matrix.png')

    print 'Plotting Test Train Error'
    error_fig = plt.figure()
    plt.plot(range(len(test_errors)), test_errors, '-r',
             label='Feature Prediction error for Test set', linewidth=1.0, ls='-')
    plt.plot(range(len(train_errors)), train_errors, '-g',
             label='Feature Prediction error for Train set', linewidth=1.0)
    plt.ylabel("Error = l2_norm( pred_feature_vector - true_feature_vector)")
    plt.xlabel("Patient Number")
    plt.title(
        "AVG Model: Feature Prediction error for patients using AVG Model")
    plt.legend(loc='upper right')
    error_fig.savefig('avg_model_figs/not_normalized/train_test_error_norm_figs')

if __name__ == '__main__':
    main()
