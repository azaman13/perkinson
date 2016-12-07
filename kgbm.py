from __future__ import division
import glob
import random
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import pylab as pl
import matplotlib.colors as colors
import seaborn as sns
import os
import shutil


def read_data(filename):
    f = open(filename)
    f.readline()
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    A = data[:, :3]
    Y = data[:, -1]
    return A, Y


def create_columns(A):
    matrix_columns = []
    for i in range(A.shape[1]):
        matrix_columns.append(A[:, i])
    # print 'Before', A.shape
    # Add square of the columns
    for vector in matrix_columns:
        new_vector = np.multiply(vector, vector)
        A = np.column_stack((A, new_vector))
        # a = 1.0 + vector
        # print a
        # c = np.log(a)
        # c = np.log(1.0 + vector)
        # A = np.column_stack((A, c))
        # A = np.column_stack((A, np.divide(1.0, 1.0 + vector)))
    for vector in matrix_columns:
        # print '**********'
        # print vector
        # break
        new_vector = np.log(np.add(2.0, vector))
        A = np.column_stack((A, new_vector))
        A = np.column_stack((A, np.divide(1.0, 2.0 + vector)))
    # print 'After', A.shape
    return A

def find_nearest_patients(A, training_files):
    results = []
    for f in training_files:
        patid = f.split('/')[-1]
        test_A, _ = read_data(f)
        distance = LA.norm(np.subtract(A, test_A))
        results.append((distance, patid))
    results.sort(key=lambda x: x[0])
    return results



def main(k):
    random.seed(5)
    path = os.environ['HOME']+'/parkinson/gbm_model_figs/'+ str(k) + '_nearest/'
    feature_path = path + 'feature_error_hists/'
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        os.mkdir(path, 0777)
        os.mkdir(path + 'feature_error_hists/', 0777)

    # list of models for each patient
    models = {}
    tolerance = 0.1

    files = glob.glob(
        "normalized_data/*.csv")

    # print len(files)
    random.shuffle(files)

    training_files = files[:100]
    testing_files = files[100:]


    # Dic of training vector. Each element is a vector representing
    # the training vector error For every training patient
    train_error_vectors = {}

    train_errors = []

    for f in training_files:

        A, Y = read_data(f)

        # print A.shape, Y.shape
        A = create_columns(A)

        # Now training
        gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,max_depth=3, random_state=0, loss='ls')
        model = gbm.fit(A, Y)

        # save the model
        models[f.split('/')[-1]] = model

        pred_y = model.predict(A)
        train_error_vector = pred_y - Y
        train_error = LA.norm(train_error_vector)

        #saving training errors
        train_error_vectors[f.split('/')[-1]] = train_error_vector
        train_errors.append(train_error)


    # Now testing
    test_errors = [] # per patient

    # This is the list of all predictions for every patient in the testing set
    # Used for creating Error Matrix
    all_prediction_vectors = []
    all_label_vectors = []

    test_percent_accurately_predicted_features = []
    error_vectors = []
    for f in testing_files:
        A, Y = read_data(f)

        k_nearest_patients = find_nearest_patients(A, training_files)
        k_nearest_models = {}
        for (d, patid) in k_nearest_patients[:k]:
            k_nearest_models[patid] = models[patid]

        A = create_columns(A)

        pred_y_per_patient = np.zeros(Y.shape)

        # take the test point and make it go through
        # all the models
        error_per_model = 0.0

        for (pat_no, model) in k_nearest_models.iteritems():
            pred_y = model.predict(A)
            error_v = pred_y - Y
            error = LA.norm(error_v)

            error_per_model += error

            pred_y_per_patient = np.add(pred_y_per_patient, pred_y)

        # Avg model prediction vector
        pred_y = 1.0 / (len(k_nearest_models)) * pred_y_per_patient
        error_vector = pred_y - Y
        test_error = LA.norm(error_vector)

        all_prediction_vectors.append(pred_y)
        all_label_vectors.append(Y)
        test_errors.append(test_error)
        error_vectors.append(error_vector)


    # We want to find how many of the elements (i.e feature scores) of the avg_error_vector
    # are within the 'tolerance' variable. Features that are inside the tolerance range
    # are correctly predicted and note which of these features are they
    feature_acuracy_per_patient = []

    for e_v in error_vectors:
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

    fig1 = plt.figure()
    plt.plot(range(len(feature_acuracy_per_patient)), feature_acuracy_per_patient, '-r', linewidth=1.0, ls='-')

    plt.ylabel("Percentage correctly predicted features")
    plt.xlabel("Patient Number")
    plt.title(
        "Percentage of features predicted correctly for patients from Test set")
    plt.savefig(path + '/'+ str(k)+'_gbm_model_feature_accuracy.png')

    fig2 = plt.figure(figsize=(12,16))
    # Error Matrix Plotting
    error_matrix = np.column_stack(error_vectors)

    # The Error Matrix Heat map
    ax = sns.heatmap(np.fabs(error_matrix), cmap="YlOrRd" )
    a = ax.get_figure()
    a.savefig(path + str(k)+'_test_error_matrix.png')


    for feature_num in range(100):
        # for all 42 patients
        fig = plt.figure()

        feature_i = error_matrix[feature_num, :]
        plt.hist(feature_i)
        plt.title("Feature error Histogram for feature number: " +
                  str(feature_num + 1))
        name = str(k)+'_feature_' + str(feature_num + 1) + '_histogram'
        fig.savefig(path + '/feature_error_hists/'+name, bbox_inches='tight')

if __name__ == '__main__':
    K = [3,5,7,9,15]
    for i in K:
        main(i)
