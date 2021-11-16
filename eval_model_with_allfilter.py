# Load in model weights and evaluate its goodness (ECE, MCE, error) also saves logits.
# ResNet model from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

from tensorflow import keras
import numpy as np
from resnet_class.resnet101 import ResNet_101
from sklearn.model_selection import train_test_split
import pandas as pd

# Imports to get "utility" package
import os

# sys.path.append(path.dirname(path.dirname(path.abspath("utility"))))
from calibration.evaluation import evaluate_model
from constants import *

if __name__ == '__main__':
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(os.path.join("results", "allfilter_trained_with_allfilter_seed.xlsx"), engine='xlsxwriter')

    for seed in SEEDS:
        #weights_file = os.path.join("model_weights", "resnet_110_10_blurdetail_u_" + str(seed) + ".h5")

        checkpoint_filepath = os.path.join("model_weights", "resnet_101_allfilter_" + str(seed),
                                           "weights-allfilter.h5")
        # load data
        results = pd.DataFrame(columns=["Filter", "Accuracy", "ECE", "MCE"])

        print("Cifar-5m evaluation")

        data_path = os.path.join("DataSets", "cifar5m_part2")
        x_filter = []
        y_filter = []
        filter_list = ["sample", "blur", "detail", "edge_enhance", "smooth", "sharp"]

        for filter in filter_list:
            filter_file = filter + "_1k_X.npy"
            filter_label = filter + "_1k_Y.npy"
            fil_data = np.load(os.path.join(data_path, filter_file))
            fil_label = np.load(os.path.join(data_path, filter_label))

            # if filter in ["blur", "detail", "edge_enhance", "smooth", "sharp"]:
            _, fil_data, _, fil_label = train_test_split(fil_data, fil_label, test_size=150,
                                                         random_state=seed)  # random_state = seed

            x_filter.append(fil_data)
            y_filter.append(fil_label)

        x_filter = np.concatenate(x_filter)
        y_filter = np.concatenate(y_filter)

        y_filter.reshape(y_filter.shape[0], 1)
        y_filter = keras.utils.to_categorical(y_filter, NUM_CLASSES)

        print("x filter data shape: ", x_filter.shape)
        print("y filter data shape: ", y_filter.shape)

        x_test, x_val, y_test, y_val = train_test_split(x_filter, y_filter, test_size=0.1,
                                                        random_state=seed)  # random_state = seed

        # img_input = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
        # output = residual_network(img_input, NUM_CLASSES, STACK_N)
        # model = Model(img_input, output)

        print("Evaluation metric for filter")
        res = evaluate_model(checkpoint_filepath, x_test, y_test, bins=15, verbose=True,
                             pickle_file="probs_resnet101_allfiler_with_allfilter_" + str(seed), x_val=x_val,
                             y_val=y_val, pickle_path="logits_with")

        res = {"Accuracy": res[0], "ECE": res[1], "MCE": res[2]}
        results = results.append(res, ignore_index=True)

        print(results)
        results.to_excel(writer, sheet_name="allfilter_cal" + str(seed), index=False)
    writer.save()
