import glob
import random
import numpy as np
from sklearn import linear_model
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
    org_data = np.genfromtxt(filename, delimiter=',')[1:]

    data = np.delete(org_data, [0, 1, 2], axis=1)
    A = data[: -1]
    Y = data[-1]
    return A, Y

def find_nearest_patients(A, training_files):
    results = []
    for f in training_files:
        patid = f.split('/')[-1]
        train_A, _ = read_data(f)

        distance = LA.norm(np.subtract(A, train_A))
        results.append((distance, patid))
    results.sort(key=lambda x: x[0])
    return results

def main(k):
    random.seed(5)

    # path = os.environ['HOME']+'/parkinson/avg_model_figs/k_nearest/not_normalized/'+ str(k) + '_nearest'
    # feature_path = path + 'feature_error_hists/'
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    # else:
    #     os.mkdir(path, 0777)
    #     os.mkdir(path + '/feature_error_hists/', 0777)


    tolerance = 0.4
    # list of models for each patient
    models = {}

    files = glob.glob("patient_wise_data/*.csv")

    random.shuffle(files)

    training_files = files[:100]
    testing_files = files[100:]

    print 'Reading Training data'

    for f in training_files:
        A, Y = read_data(f)
        reg = linear_model.Ridge(alpha=.5)
        model = reg.fit(A.transpose(), Y)
        models[f.split('/')[-1]] = model
        error = LA.norm(A.transpose().dot(model.coef_) - Y)

    # This is the list of all predictions for every patient in the
    # testing set
    all_prediction_vectors = []

    # list of all true labels
    all_label_vectors = []
    print 'Running testing...'
    # For testing

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

        testing_A, Y = read_data(f)
        # closest_model, closest_patient = find_closest_model(
        #     models, training_files, testing_A)
        k_nearest_patients = find_nearest_patients(testing_A, training_files)

        k_nearest_models = {}
        for (d, patid) in k_nearest_patients[:k]:
            k_nearest_models[patid] = models[patid]

        pred_y_per_patient = np.zeros(Y.shape)
        # take the test point and make it go through
        # all the models
        error_per_model = 0.

        for (pat_no, model) in k_nearest_models.iteritems():
            pred_y = A.transpose().dot(model.coef_)
            error_v = pred_y - Y
            error = LA.norm(error_v)
            error_per_model += error

            pred_y_per_patient = np.add(pred_y_per_patient, pred_y)
        pred_y = 1.0 / (len(k_nearest_models)) * pred_y_per_patient
        error_vector = pred_y - Y
        test_error = LA.norm(error_vector)
        test_errors.append(test_error)
        error_vectors.append(error_vector)
    # We want to find how many of the elements (i.e feature scores)
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
    plt.title("AVG "+str(k) +" Closest patient: Percentage of features predicted correctly for patients from Test set")
    plt.savefig('avg_model_figs/k_nearest/not_normalized' + '/'+ str(k)+'_avg_mode_feature_accuracy.png')
    print 'Plotting Heatmap'
    fig2 = plt.figure(figsize=(12,16))
    # Error Matrix Plotting
    error_matrix = np.column_stack(error_vectors)

    # The Error Matrix Heat map
    ax = sns.heatmap(np.fabs(error_matrix), cmap="YlOrRd", xticklabels=False)
    ax.set(xlabel='patient id', ylabel='feature number', title='Error Matix on Test set for '+str(k)+' nearest patient')
    a = ax.get_figure()
    a.savefig('avg_model_figs/k_nearest/not_normalized/' + str(k)+'_test_error_matrix.png')

    # plotting error Histograms
    #### FIX THIS PART
    # for feature_num in range(42):
    #     # for all 42 patients
    #     fig = plt.figure()
    #
    #     feature_i = error_matrix[feature_num, :]
    #     plt.hist(feature_i)
    #     plt.title("Feature error Histogram for feature number: " +
    #               str(feature_num + 1))
    #     name = str(k)+'_feature_' + str(feature_num + 1) + '_histogram'
    #     fig.savefig(path + '/feature_error_hists/'+name, bbox_inches='tight')



if __name__ == '__main__':
    K = [2, 3,5,7,9, 11, 15]
    for i in K:
        main(i)
