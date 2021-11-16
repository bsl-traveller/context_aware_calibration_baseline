import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import sys
import sklearn.metrics as metrics
import pickle
#from tensorflow.nn import softmax

from pathlib import Path
# setting path
sys.path.append('..')
from constants import *
from resnet_class.resnet101 import ResNet_101

class calibrationError:
    def __init__(self, conf, pred, ground_truth, bin_size=0.1):
        '''
        Class to calculate Calibration Errors and bin info

        Args:
            conf (numpy.ndarray): list of confidences
            pred (numpy.ndarray): list of predictions
            ground_truth (numpy.ndarray): list of true labels
            bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        '''
        self.conf = conf
        self.pred = pred
        self.ground_truth = ground_truth
        self.bin_size = bin_size

        self.ECE = None
        self.MCE = None
        self.BIN_INFO = dict()

    def _compute_acc_bin(self, conf_thresh_lower, conf_thresh_upper):
        '''
        # Computes accuracy and average confidence for bin

        Args:
            conf_thresh_lower (float): Lower Threshold of confidence interval
            conf_thresh_upper (float): Upper Threshold of confidence interval

        Returns:
            (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
        '''

        filtered_tuples = [x for x in zip(self.pred, self.ground_truth, self.conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
        if len(filtered_tuples) < 1:
            return 0, 0, 0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct) / len_bin  # accuracy of BIN
            return accuracy, avg_conf, len_bin


    def calculate_errors(self):

        upper_bounds = np.arange(self.bin_size, 1 + self.bin_size, self.bin_size)  # Get bounds of bins

        n = len(self.conf)
        ece = 0  # Starting error
        cal_errors = []
        accuracies = []
        confidences = []
        bin_lengths = []

        for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
            acc, avg_conf, len_bin = self._compute_acc_bin(conf_thresh - self.bin_size, conf_thresh)
            accuracies.append(acc)
            confidences.append(avg_conf)
            bin_lengths.append(len_bin)
            ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE
            cal_errors.append(np.abs(acc - avg_conf))

        self.ECE = ece
        self.MCE = max(cal_errors)
        self.BIN_INFO["accuracies"] = accuracies
        self.BIN_INFO["confidences"] = confidences
        self.BIN_INFO["bin_lengths"] = bin_lengths

        return self.ECE, self.MCE, self.BIN_INFO


def softmax(x, axis = 1):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    m = np.max(x, axis=axis)
    m = m[:, np.newaxis]
    e_x = np.exp(x - m)
    #print("m shape: ", m.shape)
    #print("e_x shape: ", e_x.shape)
    return e_x / e_x.sum(axis=axis, keepdims=1)


def evaluate_model(weights_file, x_test, y_test, bins=15, verbose=True, pickle_file=None, x_val=None,
                   y_val=None, pickle_path=None, contexts=False):
    """
    Evaluates the model, in addition calculates the calibration errors and
    saves the logits for later use, if "pickle_file" is not None.

    Parameters:
        weights_file: (string): path to weights file
        x_test: (numpy.ndarray) with test data
        y_test: (numpy.ndarray) with test data labels
        bins: Number of bins
        verbose: (boolean) print out results or just return these
        pickle_file: (string) path to pickle probabilities given by model
        x_test: (numpy.ndarray) with validation data
        y_test: (numpy.ndarray) with validation data labels
        pickle_path: Store location of pickle file

    Returns:
        (acc, ece, mce): accuracy of model, ECE and MCE (calibration errors)
    """

    # Change last activation to linear (instead of softmax)
    img_shape = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
    model = ResNet_101(img_shape, NUM_CLASSES, WEIGHT_DECAY, last_layer_activation='linear', contexts=contexts)


    # First load in the weights
    model.load_weights(weights_file)  # .expect_partial()
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    # Next get predictions
    y_logits = model.predict(x_test)
    if contexts:
        y_logits = y_logits[:, :10]
    y_probs = softmax(y_logits, axis=1)
    y_preds = np.argmax(y_probs, axis=1)
    y_true = y_test

    # Find accuracy and error
    if y_true.shape[1] > 1:  # If 1-hot representation, get back to numeric
        y_true = np.array([[np.where(r == 1)[0][0]] for r in y_true])  # Back to np array also

    accuracy = metrics.accuracy_score(y_true, y_preds) * 100
    error = 100 - accuracy

    # Confidence of prediction
    y_confs = np.max(y_probs, axis=1)  # Take only maximum confidence

    errors = calibrationError(y_confs, y_preds, y_true, bin_size=1 / bins)
    ece, mce, _ = errors.calculate_errors()

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)

    # Pickle probabilities for test and validation
    if pickle_file:

        # Get predictions also for x_val
        y_logits_val = model.predict(x_val)
        if contexts:
            y_logits_val = y_logits_val[:, :10]
        y_probs_val = softmax(y_logits_val, axis=1)
        y_preds_val = np.argmax(y_probs_val, axis=1)

        #
        if y_val.shape[1] > 1:  # If 1-hot representation, get back to numeric
            y_val = np.array([[np.where(r == 1)[0][0]] for r in y_val])  # Also convert back to np.array, TODO argmax?

        if verbose:
            print("Pickling the probabilities for validation and test.")
            print("Validation accuracy: ", metrics.accuracy_score(y_val, y_preds_val) * 100)

        # Write file with pickled data

        Path(pickle_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(pickle_path, pickle_file + '_logits.p'), 'wb') as f:
            pickle.dump([(y_logits_val, y_val), (y_logits, y_true)], f)

    # Return the basic results
    return (accuracy, ece, mce)








