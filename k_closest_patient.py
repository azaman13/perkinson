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
    NO_MODEL = True
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
    # Now testing
    test_errors = [] # per patient

    # This is the list of all predictions for every patient in the testing set
    # Used for creating Error Matrix
    all_prediction_vectors = []
    all_label_vectors = []

    test_percent_accurately_predicted_features = []
    error_vectors = []
    for f in testing_files:
        A_test, Y_test = read_data(f)

        # take the k closest patients
        k_nearest_patients = find_nearest_patients(A_test, training_files)[:k]

        pred_y_per_patient = np.zeros(Y_test.shape)
        error_vector_per_patient = np.zeros(Y_test.shape)

        for (distance, patid) in k_nearest_patients[:k]:
            # now read patid file
            filename = 'normalized_data/'+patid
            closest_A, closest_Y = read_data(filename)
            # get Y and use it to compute error by treating it
            # as the label
            error_vector = closest_Y - Y_test

            pred_y_per_patient = np.add(pred_y_per_patient, closest_Y)
            error_vector_per_patient = np.add(error_vector_per_patient, error_vector)

        # avg predictions
        pred_y = (1.0 / k) * pred_y_per_patient
        error_vector =  (1.0 / k) * error_vector_per_patient
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
    name = 'clustering_figs/avg/'+ str(k) + '_avg_closest.png'
    plt.savefig(name)

    fig2 = plt.figure(figsize=(12,16))

    # Error Matrix Plotting
    error_matrix = np.column_stack(error_vectors)
    # The Error Matrix Heat map
    ax = sns.heatmap(np.fabs(error_matrix), cmap="YlOrRd" )
    a = ax.get_figure()
    name = 'clustering_figs/avg/'+ str(k) + '_avg_closest_error_matrix.png'
    a.savefig(name)


if __name__ == '__main__':
    K = [1,2,3,5,7,9,15]
    for i in K:
        main(i)
